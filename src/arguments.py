import argparse

#Checkout https://github.com/awslabs/sockeye/blob/master/sockeye/arguments.py and https://stackoverflow.com/questions/28579661/getting-required-option-from-namespace-in-python

def add_arguments(parser):
    # Model parameters
    parser.add_argument("--event_vector_size",type=int,help="Size of each of the event vectors")
    parser.add_argument("--rnn_hidden_size",type=int,help="Hidden size of the RNN layers")
    parser.add_argument("--classifier_hidden_size",type=int,help="Size of the classifier linear layers")
    
    
    # Problem parameters
    parser.add_argument("--n_players",type=int,help="Number of players in the game")
    parser.add_argument("--n_roles",type=int,help="Number of possible roles (classes)")


    # Training
    parser.add_argument("--update_frequency",type=int,help="Gradient accomulation.Simulates batch size")
    parser.add_argument("--loss_scale",type=str,choices=["last_step_only","uniform","linearly_growing"],help="Type of loss used for training the model.")
    parser.add_argument("--log_frequency",type=int,default=100,help="Show training statistics every log_frequency updates")
    parser.add_argument("--gpu_device_number",type=int,required=False,default=-1,help="Ordinal device number of gpu. Negative number means use CPU")
    
