"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union, overload

T = TypeVar("T", bound="SAE")
import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae_lens.config import DTYPE_MAP
from sae_lens.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.toolkit.pretrained_saes_directory import (
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
)

from huggingface_hub import hf_hub_download
import safetensors.torch
from typing import NamedTuple, Optional
import os

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"

@dataclass
class SAEConfig:
    # architecture details
    architecture: Literal["standard", "topk"]
    # forward pass details.
    k: int
    d_in: int
    d_sae: int
    activation_fn_str: str
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool
    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str
    dataset_trust_remote_code: bool
    normalize_activations: str

    # misc
    dtype: str
    device: str
    sae_lens_training_version: Optional[str]
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    neuronpedia_id: Optional[str] = None
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
        # rename dict:
        rename_dict = {  # old : new
            "hook_point": "hook_name",
            "hook_point_head_index": "hook_head_index",
            "hook_point_layer": "hook_layer",
            "activation_fn": "activation_fn_str",
        }
        config_dict = {rename_dict.get(k, k): v for k, v in config_dict.items()}
        # use only config terms that are in the dataclass
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k in cls.__dataclass_fields__  # pylint: disable=no-member
        }
        return cls(**config_dict)

    # def __post_init__(self):

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "activation_fn_str": self.activation_fn_str,  # use string for serialization
            "activation_fn_kwargs": self.activation_fn_kwargs or {},
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
            "neuronpedia_id": self.neuronpedia_id,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs,
        }

class SAE(HookedRootModule):
    """
    Core Sparse Autoencoder (SAE) class used for inference. For training, see `TrainingSAE`.
    """
    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device
    # analysis
    use_error_term: bool
    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        if cfg.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.activation_fn = get_activation_fn(
            cfg.activation_fn_str, **cfg.activation_fn_kwargs or {}
        )
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        if self.cfg.architecture == "standard":
            self.initialize_weights_basic()
            self.encode = self.encode_standard
        elif self.cfg.architecture == "topk":
            self.initialize_weights_basic()
            self.encode = self.encode_topk
        else:
            raise (ValueError)

        # handle presence / absence of scaling factor.
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint() # after encode SAE (not included self.activation_fn)
        self.hook_sae_acts_post = HookPoint() # after encoder SAE (included self.activation_fn)
        self.hook_sae_acts_topk = HookPoint() # topk activations after encode SAE in case of topk architecture
        self.hook_sae_output = HookPoint() # output, after decode SAE
        
        self.hook_sae_recons = HookPoint() # for analysis
        self.hook_sae_error = HookPoint() # for analysis
        if self.cfg.hook_name.endswith("_z"):
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # need to default the reshape fns
            self.turn_off_forward_pass_hook_z_reshaping()
        
        if self.cfg.normalize_activations == "constant_norm_rescale":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                x = x * self.x_norm_coeff
                return x
            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:  #
                x = x / self.x_norm_coeff
                del self.x_norm_coeff  # prevents reusing
                return x
            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x
            def run_time_activation_ln_out(x: torch.Tensor, eps: float = 1e-5):
                return x * self.ln_std + self.ln_mu
            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x
        
        self.setup()  # Required for `HookedRootModule`s
        # handle run time activation normalization if needed:

    def initialize_weights_basic(self):
        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.cfg.finetuning_scaling_factor:
            self.finetuning_scaling_factor = nn.Parameter(
                torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )

    @overload
    def to(
        self: T,
        device: Optional[Union[torch.device, str]] = ...,
        dtype: Optional[torch.dtype] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args: Any, **kwargs: Any) -> "SAE":  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the .device property
            self.device = device

        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the .dtype property
            self.dtype = dtype_arg

        # Call the parent class's to() method to handle all cases (device, dtype, tensor)
        return super().to(*args, **kwargs)

    # Basic Forward Pass Functionality.
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)
        return self.hook_sae_output(sae_out)
    
    def encode_topk(
        self,
        x: torch.Tensor,
    ):
        x = x.to(self.dtype)
        sae_in = self.reshape_fn_in(x)
        # sae_in = self.run_time_activation_norm_fn_in(sae_in)
        sae_in_cent = sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)
        # Encoding
        pre_acts = self.hook_sae_acts_pre(sae_in_cent @ self.W_enc + self.b_enc)
        post_acts = self.activation_fn(pre_acts) # TODO: to decide using activation function or not
        # Top-k selection
        if post_acts.shape[-1] >= self.cfg.k:
            # warning, we are testing less than k features
            top_acts, top_indices = post_acts.topk(self.cfg.k, dim=-1, sorted=False)
            top_k_acts = torch.zeros_like(post_acts) # [bs, seq_len, d_sae]
            for i in range(top_indices.shape[0]):
                for j in range(top_indices.shape[1]):
                    top_k_acts[i, j, top_indices[i][j]] = top_acts[i][j]
        else:
            warnings.warn("Less than k features in the input. Testing feature-centric only.")
            top_k_acts = post_acts

        top_k_acts = self.hook_sae_acts_post(top_k_acts) # record top_k activations
        return top_k_acts            

    def encode_standard(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """

        x = x.to(self.dtype)
        x = self.reshape_fn_in(x)
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)
        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(
            self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
        )
        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore
        return sae_out

    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        if self.cfg.architecture == "gated":
            self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
            self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
            self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()
        else:
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: float
    ):
        self.W_enc.data = self.W_enc.data * activation_norm_scaling_factor
        # previously weren't doing this.
        self.W_dec.data = self.W_dec.data / activation_norm_scaling_factor

        # once we normalize, we shouldn't need to scale activations.
        self.cfg.normalize_activations = "none"

    def save_model(self, path: str, sparsity: Optional[torch.Tensor] = None):

        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = self.cfg.to_dict()

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "SAE":

        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
            dtype=DTYPE_MAP[cfg_dict["dtype"]],
        )

        sae_cfg = SAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["SAE", dict[str, Any], Optional[torch.Tensor]]:
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model directory in the Hugging Face model hub.
        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # get the repo id and path to the SAE
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # If using Gemma Scope and not the canonical release, give a hint to use it
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )
        sae_info = sae_directory.get(release, None)
        hf_repo_id = sae_info.repo_id if sae_info is not None else release
        hf_path = sae_info.saes_map[sae_id] if sae_info is not None else sae_id
        config_overrides = sae_info.config_overrides if sae_info is not None else None
        neuronpedia_id = (
            sae_info.neuronpedia_id[sae_id] if sae_info is not None else None
        )

        conversion_loader_name = "sae_lens"
        if sae_info is not None and sae_info.conversion_func is not None:
            conversion_loader_name = sae_info.conversion_func
        if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
            raise ValueError(
                f"Conversion func {conversion_loader_name} not found in NAMED_PRETRAINED_SAE_LOADERS."
            )
        conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=hf_repo_id,
            folder_name=hf_path,
            device=device,
            force_download=False,
            cfg_overrides=config_overrides,
        )

        sae = cls(SAEConfig.from_dict(cfg_dict))
        sae.load_state_dict(state_dict)
        sae.cfg.neuronpedia_id = neuronpedia_id

        # Check if normalization is 'expected_average_only_in'
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities
    
    
    @classmethod
    def from_eleuther(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
        decoder: bool = True,  # Default to True as in implementation 1
    ) -> "SAE":
        """
        Load a model saved by implementation 1 into an SAE instance in implementation 2.

        Args:
            path (str): Path to the directory containing 'sae.safetensors' and 'cfg.json'.
            device (str): Device to load the model onto.
            decoder (bool): Whether to initialize the decoder. Should match how the model was saved in implementation 1.

        Returns:
            SAE: An instance of SAE loaded with weights from implementation 1.
        """
        # Load the state_dict from the safetensors file saved by implementation 1
        
        sae_path = hf_hub_download(
            repo_id=release, filename=f"{sae_id}/sae.safetensors"
        )
        cfg_path = hf_hub_download(
            repo_id=release, filename=f"{sae_id}/cfg.json", force_download=True
        )
        state_dict = safetensors.torch.load_file(
            sae_path, device=device
        )
        # Map the parameter names and adjust shapes as necessary
        new_state_dict = {}
        new_state_dict["W_enc"] = state_dict["encoder.weight"].T  # Transpose to match shape
        new_state_dict["b_enc"] = state_dict["encoder.bias"]
        new_state_dict["W_dec"] = state_dict["W_dec"]
        new_state_dict["b_dec"] = state_dict["b_dec"]

        # Load the configuration from cfg.json
        print("-----Loading from eleuther-----")
        print(cfg_path)
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        print(cfg_dict)
        cfg_dict["layer"] = int(sae_id.split(".")[-1])
        # Adjust the cfg_dict if necessary to match SAEConfig fields
        # For example, rename fields or set default values
        cfg_dict = cls.adjust_cfg_dict(cfg_dict)
        # Create an SAEConfig object
        sae_config = SAEConfig.from_dict(cfg_dict)

        # Initialize the SAE model
        sae = cls(sae_config)

        # Load the adjusted state_dict
        sae.load_state_dict(new_state_dict)

        return sae, cfg_dict, None
    
    
    @staticmethod
    def adjust_cfg_dict(cfg_dict):
        """
        Adjust the cfg_dict from implementation 1 to match the fields expected by SAEConfig in implementation 2.
        """
        # Set default values for any missing fields required by SAEConfig
        default_values = {
            'architecture': 'standard',  # Default architecture
            'd_in': cfg_dict["d_in"],  # Set appropriate default
            'activation_fn_str': 'relu',
            'apply_b_dec_to_input': False,
            'finetuning_scaling_factor': False,
            'context_size': 1024,  # Set appropriate default
            'model_name': '',  # Set appropriate default
            'hook_name': f'blocks.{cfg_dict["layer"]}.hook_res',  # Set appropriate default
            'hook_layer': cfg_dict["layer"],  # Set appropriate default
            'hook_head_index': None,
            'prepend_bos': False,
            'dataset_path': '',
            'dataset_trust_remote_code': False,
            'normalize_activations': 'none',
            'dtype': 'float32',
            'device': 'cpu',
            'sae_lens_training_version': None,
            'activation_fn_kwargs': {'k': cfg_dict["k"]},
            'neuronpedia_id': None,
            'model_from_pretrained_kwargs': {},
            
        }
        for key, value in default_values.items():
            cfg_dict.setdefault(key, value)
        print(cfg_dict)
        return cfg_dict

    def get_name(self):
        sae_name = f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"
        return sae_name

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))

    def turn_on_forward_pass_hook_z_reshaping(self):

        assert self.cfg.hook_name.endswith(
            "_z"
        ), "This method should only be called for hook_z SAEs."

        def reshape_fn_in(x: torch.Tensor):
            self.d_head = x.shape[-1]  # type: ignore
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in

        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x
        self.d_head = None
        self.hook_z_reshaping_mode = False


class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            input = torch.tanh(input)
            return input

        return tanh_relu
    elif activation_fn == "topk":
        assert "k" in kwargs, "TopK activation function requires a k value."
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided

        return TopK(k, postact_fn)
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")
