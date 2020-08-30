from google.cloud import storage
from hyperpose.Config import config_opps
import tensorflow as tf
import os
storage_client = storage.Client()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)


def gcs_file_exists(bucket_name, file_path):
    bucket = storage_client.bucket(bucket_name)
    return storage.Blob(bucket=bucket, name=file_path).exists(storage_client)


def get_tpu_resolver():

    print("All devices: ", tf.config.list_logical_devices('TPU'))
    return resolver
