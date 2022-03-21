def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--logging_enabled', action='store_true', help='learning rate for training')
    parser.add_argument('--resume_training', action='store_true', help='whether to resume training based on a hardcoded saved model path')
    parser.add_argument('--config', type=str, help='path to .yaml config file')

    return parser
