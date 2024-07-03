import argparse
import nugraph as ng

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
	parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def loss_landscape(args):

	nudata = Data(data_path=args.data_path)
    

if __name__ == '__main__':
	args = configure()
	loss_landscape(args)