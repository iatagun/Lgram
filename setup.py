from setuptools import setup, find_packages

setup(
    name='lgram',  # Paketinizin adı
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'spacy',  # Bağımlı kütüphaneler
    ],
    description='A centering theory-based text coherence analysis package.',
    author='İlker Atagün',
    author_email='ilker.atagun@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
