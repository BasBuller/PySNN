Installation
============

Installation can be done with pip:

.. code-block:: bash

    $ pip install pysnn

If you want to make updates to the library without having to reinstall it, use the following commands instead:

.. code-block:: bash

    $ git clone https://github.com/BasBuller/PySNN.git
    $ cd PySNN/
    $ pip install -e PySNN/

Some examples need additional libraries. To install these, run:

.. code-block:: bash

    $ pip install pysnn[examples]

Code is formatted with `Black <https://github.com/psf/black>`_ using a pre-commit hook. To configure it, run:

.. code-block:: bash

    $ pre-commit install
