Installation
============

Bionium-X requires **Python 3.9+** and can be installed via ``pip``.
We recommend using a virtual environment (e.g., ``venv`` or ``conda``).

Core Installation
-----------------

The core library is lightweight and installs in seconds. It relies solely on standard scientific packages (``numpy``, ``scipy``, ``astropy``).

.. code-block:: bash

    pip install bionium-x

*Note: If installing from the source repository:*

.. code-block:: bash

    git clone https://github.com/bionium-x-research/Bionium-X.git
    cd Bionium-X
    pip install .

Optional Dependencies
---------------------

Bionium-X offers several optional extensions for heavy computational tasks.

Machine Learning (``[ml]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
To utilize the 1D Convolutional Neural Network (CNN) for rapid spectral inference, you need to install the PyTorch dependencies:

.. code-block:: bash

    pip install bionium-x[ml]

This will download the heavy PyTorch binaries and allow ``bioniumx.models.fetch`` to download the pre-trained `.pth` network weights.

Advanced Science (``[science]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For Bayesian model comparison, Markov Chain Monte Carlo (MCMC) fitting, and parallelized physics modules:

.. code-block:: bash

    pip install bionium-x[science]

Full Installation
^^^^^^^^^^^^^^^^^
To install all recommended extensions:

.. code-block:: bash

    pip install bionium-x[full]
