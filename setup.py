from distutils.core import setup, Extension
import numpy

def main():
    
    setup(name="deepvi",
          version="1.0.0",
          description="Library of architectures for various applications of variational inference using deep learning",
          author="Clayton Seitz",
          author_email="cwseitz@iu.edu")



if __name__ == "__main__":
    main()
