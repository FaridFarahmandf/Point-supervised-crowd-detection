"""
Script to download ResNet-50 weights for the project
"""
import os
import urllib.request
import sys

def download_file(url, destination):
    """Download a file from URL to destination"""
    print('Downloading {}...'.format(url))
    print('Saving to: {}'.format(destination))
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write('\rProgress: {}%'.format(percent))
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print('\nDownload completed successfully!')
        return True
    except Exception as e:
        print('\nError downloading file: {}'.format(e))
        return False

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    models_dir = 'data/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print('Created directory: {}'.format(models_dir))
    
    # Download URL
    url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    destination = os.path.join(models_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    
    # Check if file already exists
    if os.path.exists(destination):
        print('File already exists at: {}'.format(destination))
        response = input('Do you want to download again? (y/n): ')
        if response.lower() != 'y':
            print('Skipping download.')
            sys.exit(0)
    
    # Download the file
    if download_file(url, destination):
        print('File saved to: {}'.format(destination))
        file_size = os.path.getsize(destination) / (1024 * 1024)  # Size in MB
        print('File size: {:.2f} MB'.format(file_size))
    else:
        print('Failed to download file. Please download manually from:')
        print(url)
        sys.exit(1)

