#!/usr/bin/env python
import argparse
import os
import torch
import nugraph as ng

# Assign your model class to Model
Model = ng.models.NuGraph3

def configure():
    parser = argparse.ArgumentParser(description='Convert model to TorchScript.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the TorchScript model')
    return parser.parse_args()

def convert_to_torchscript(checkpoint_path, output_path):
    # Load the model from checkpoint
    print(f'Loading model from checkpoint: {checkpoint_path}')
    model = Model.load_from_checkpoint(checkpoint_path)
    
    # Convert to TorchScript
    script = model.to_torchscript()
    
    # Save the TorchScript model
    output_file_path = os.path.join(output_path, "model.pt")
    print(f'Saving TorchScript model to: {output_file_path}')
    torch.jit.save(script, output_file_path)

if __name__ == '__main__':
    args = configure()
    convert_to_torchscript(args.checkpoint, args.output)

