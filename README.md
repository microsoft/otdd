# Optimal Transport Dataset Distance (OTDD)

## Getting Started

### Installation

Note: It is highly recommended that the following be done inside a virtual environment

First install dependencies. Start by install pytorch with desired configuration using the instructions [here](https://pytorch.org/get-started/locally/).

Then do:
```
pip install -r requirements.txt
# local package
-e .
```
Then this package:
```
pip install -e .
```

### Usage Examples

A vanilla example:

```python
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance


# Load data etc ...
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          precision='single',                          
                          p = 2, entreg = 1e-1,
                          device=device)

d = dist.distance(maxsamples = 1000)

```

## Advanced Usage

### Using a custom feature distance

By default, OTDD using the (squared) Euclidean distance between features. To use a custom distance in domains where it makes sense to use one (e.g., images), one can pass a callable to OTDD using the `feature_cost` arg. Example:

```python

from otdd.pytorch.distance import DatasetDistance, embedded_feature_cost
from torchvision.models import resnet18
from otdd.pytorch.utils import load_torchvision_data
from functools import partial

# Load MNIST/CIFAR in 3channels (needed by torchvision models)

loaders_src,_ = load_torchvision_data('CIFAR10',  valid_size=0,  resize = 28, maxsize=2000)
loaders_tgt,_ = load_torchvision_data('MNIST', valid_size=0, resize = 28,
                                      to3channels=True, maxsize=2000) # No splitting at first


# Embed using a pretrained (+frozen) resnet
embedder = resnet18(pretrained=True).eval()
embedder.fc = torch.nn.Identity()
for p in embedder.parameters():
    p.requires_grad = False

# Here we use same embedder for both datasets
feature_cost = partial(embedded_feature_cost,
                           emb_x = embedder,
                           dim_x = (3,28,28),
                           emb_y = embedder,
                           dim_y = (3,28,28),
                           p = 2,
                           device=device)

dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = feature_cost,
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          precision='single',                          
                          p = 2, entreg = 1e-1,
                          device=device)

d = dist.distance(maxsamples = 10000)

```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
