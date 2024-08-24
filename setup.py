from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ai-text-moderator',
    version='0.1',
    author='Dante Capobianco',
    author_email='capodevservices@gmail.com',
    description='Content moderation transformer model classifying text across 7 labels',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Dante-Capobianco/ai-text-moderator',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

