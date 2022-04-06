from setuptools import setup

setup(name="number_classifier_kNN",
        version="1.0.0",
        description="Classifier for numbers in MNist database",
        packages=[],
        entry_points={
            "console_scripts" : ["mnist_classifier = classifier.main:run"]},
        install_requires=["python-mnist"],
        zip_safe = False        
        )
