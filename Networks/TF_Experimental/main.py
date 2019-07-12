import tensorflow as tf
import tf_linreg
import tf_linreg_example
import tf_v1_binary_class
import tf_v2_binary_class
import tf_v3_binary_class


def main():
    print(tf.__version__)
    # tf_linreg.linreg_main()
    # tf_linreg2.linreg2_main()
    # tf_linreg_example.linreg_example_main()
    # tf_v2_binary_class.bin_classifier_main()
    tf_v3_binary_class.run()


if __name__ == "__main__":
    main()
