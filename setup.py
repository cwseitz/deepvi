from distutils.core import setup, Extension
import os

def main():
    
    requirement_path = 'requirements.txt'
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requires = f.read().splitlines()
    print(install_requires)
    setup(name="deepvi",
          version="1.0.0",
          description="Library of architectures for various applications of variational inference using deep learning",
          author="Clayton Seitz",
          install_requires=install_requires,
          author_email="cwseitz@iu.edu")



if __name__ == "__main__":
    main()
