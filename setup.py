from setuptools import setup, find_packages


setup(
    name="ws_crl",
    version="1.0.0",
    packages=find_packages(include=["ws_crl*", "experiments"]),
    scripts=[
        "experiments/causalcircuit_reduce_dim.py",
        "experiments/scaling.py",
        "experiments/causalcircuit.py",
    ],
    url="",
    license='BSD 3-clause "Clear" License',
    author="Johann Brehmer and Pim de Haan",
    author_email="jbrehmer@qti.qualcomm.com",
    description="Code for Weakly Supervised Causal Representation Learning",
)
