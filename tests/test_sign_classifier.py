import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.model_registry import ModelRegistry
from models.sign_classifier import SignClassifier


class SignClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_softmax_output_and_logits_share_weights_and_have_gradients(self):
        model = SignClassifier(class_count=3, pretrained=False, hidden_size=8)
        model.eval()
        inputs = torch.rand(2, 3, 64, 64)
        probabilities = model(inputs)
        logits = model.forward_logits(inputs)
        self.assertIsInstance(model.softmax, nn.Softmax)
        torch.testing.assert_close(probabilities, logits.softmax(dim=1))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
        nn.CrossEntropyLoss()(logits, torch.tensor([0, 2])).backward()
        self.assertIsNotNone(model.network.fc[4].weight.grad)
        self.assertIsNotNone(model.network.layer4[0].conv1.weight.grad)
        self.assertIsNone(model.network.conv1.weight.grad)
        # Softmax não introduz pesos novos: checkpoints da antiga cabeça continuam válidos.
        self.assertFalse(any(name.startswith("softmax") for name in model.state_dict()))

    def test_hidden_layers_are_configurable_and_frozen_backbone_stays_in_eval(self):
        model = SignClassifier(
            class_count=4, pretrained=False, hidden_size=12, num_layers=3,
            dropout=0.3, trainable_layers=("fc",),
        )
        linears = [layer for layer in model.network.fc if isinstance(layer, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linears], [(512, 12), (12, 12), (12, 12), (12, 4)])
        model.train()
        self.assertTrue(model.network.fc.training)
        self.assertFalse(model.network.layer4.training)
        self.assertTrue(all(name.startswith("network.fc.") for name, parameter in model.named_parameters() if parameter.requires_grad))
        model.eval()
        probabilities = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(probabilities.shape, (2, 4))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_invalid_architecture_has_a_clear_error(self):
        for options in ({"num_layers": 0}, {"hidden_size": 0}, {"dropout": 1}):
            with self.assertRaises(ValueError):
                SignClassifier(class_count=3, pretrained=False, **options)

    def test_registry_keeps_generic_logits_and_requires_logits_method_for_probabilities(self):
        class GenericModel(nn.Module):
            INPUT_FORMAT = "features"

            def __init__(self, class_count):
                super().__init__()
                self.layer = nn.Linear(2, class_count)

            def forward(self, inputs):
                return self.layer(inputs)

        class InvalidModel(GenericModel):
            OUTPUT_FORMAT = "probabilities"

        with patch.dict(ModelRegistry.MODELS, {"generic": GenericModel, "invalid": InvalidModel}):
            self.assertIsInstance(ModelRegistry.create("generic", 3, {}), GenericModel)
            with self.assertRaisesRegex(ValueError, "forward_logits"):
                ModelRegistry.create("invalid", 3, {})
        self.assertNotIn("landmark_lstm", ModelRegistry.names())


if __name__ == "__main__":
    unittest.main()
