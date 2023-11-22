from setuptools import setup

setup(
    name="lama",
    install_requires=['tqdm', 'pandas', 'urllib3'],
    version='1.2',
    description="All necessary tools for the LAMA student lab at ITIV",
    url="https://www.itiv.kit.edu/60_LAMA.php",
    packages=['lama'],
)
