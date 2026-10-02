import importlib
import inspect
from pathlib import Path

from torch import nn

from models.landmark_lstm import LandmarkLSTM
from models.sign_classifier import SignClassifier


class ModelRegistry():
    MODELS = {
        "resnet18": SignClassifier,
        "landmark_lstm": LandmarkLSTM,
    }

    # Público
    @classmethod
    def create(cls, name: str, class_count: int, parameters: dict) -> nn.Module:
        model_class = cls.get_class(name)
        signature = inspect.signature(model_class)
        allowed_parameters = {
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name != "class_count"
            and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        }
        accepts_keywords = any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        unknown_parameters = set(parameters) - allowed_parameters
        if "class_count" in parameters or (unknown_parameters and not accepts_keywords):
            raise ValueError(
                f"Parâmetros desconhecidos para '{name}': {', '.join(sorted(unknown_parameters))}. "
                f"Argumentos disponíveis: {', '.join(sorted(allowed_parameters))}. "
                "class_count é definido pelo manifesto."
            )

        return model_class(class_count=class_count, **parameters)

    @classmethod
    def get_class(cls, name: str) -> type[nn.Module]:
        if name in cls.MODELS:
            return cls.MODELS[name]

        if ":" not in name:
            raise ValueError(
                f"Modelo desconhecido: '{name}'. Disponíveis: {', '.join(cls.names())}. "
                "Para uma classe externa, use pacote.modulo:Classe."
            )

        module_name, class_name = name.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name, None)
        if not inspect.isclass(model_class) or not issubclass(model_class, nn.Module):
            raise ValueError(f"'{name}' precisa indicar uma classe que herda torch.nn.Module.")
        return model_class

    @classmethod
    def input_format(cls, name: str) -> str:
        model_class = cls.get_class(name)
        input_format = getattr(model_class, "INPUT_FORMAT", None)
        if not isinstance(input_format, str) or not input_format:
            raise ValueError(f"O modelo '{name}' deve definir INPUT_FORMAT, por exemplo 'image' ou 'sequence'.")
        return input_format

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.MODELS)

    @classmethod
    def source_path(cls, name: str) -> Path:
        return Path(inspect.getfile(cls.get_class(name))).resolve()
