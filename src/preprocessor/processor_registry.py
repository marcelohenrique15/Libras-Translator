import importlib
import inspect
from pathlib import Path

from preprocessor.landmark_processor import LandmarkProcessor


class ProcessorRegistry():
    PROCESSORS = {
        "landmarks": LandmarkProcessor,
    }

    # Público
    @classmethod
    def create(cls, name: str, root: Path, parameters: dict):
        processor_class = cls.get_class(name)
        signature = inspect.signature(processor_class)
        allowed_parameters = {
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name != "dataset_root"
            and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        }
        accepts_keywords = any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        unknown_parameters = set(parameters) - allowed_parameters
        if "dataset_root" in parameters or (unknown_parameters and not accepts_keywords):
            raise ValueError(
                f"Parâmetros desconhecidos para '{name}': {', '.join(sorted(unknown_parameters))}. "
                f"Argumentos disponíveis: {', '.join(sorted(allowed_parameters))}."
            )
        return processor_class(root, **parameters)

    @classmethod
    def source_path(cls, name: str) -> Path:
        return Path(inspect.getfile(cls.get_class(name))).resolve()

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.PROCESSORS)

    @classmethod
    def get_class(cls, name: str):
        if name in cls.PROCESSORS:
            return cls.PROCESSORS[name]
        if ":" not in name:
            raise ValueError(
                f"Processor desconhecido: '{name}'. Disponíveis: {', '.join(cls.names())}. "
                "Para uma classe externa, use pacote.modulo:Classe."
            )

        module_name, class_name = name.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        processor_class = getattr(module, class_name, None)
        if not inspect.isclass(processor_class):
            raise ValueError(f"'{name}' precisa indicar uma classe de processamento.")
        return processor_class
