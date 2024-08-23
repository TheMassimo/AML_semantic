class MyArgs:
    def __init__(self,
                mode = 'train',
                backbone = 'CatmodelSmall',
                pretrain_path = '',
                use_conv_last = False,
                num_epochs = 5,
                epoch_start_i = 0,
                checkpoint_step = 10,
                validation_step = 1,
                crop_height = 512,
                crop_width = 1024,
                batch_size = 2,
                learning_rate = 0.01,
                num_workers = 0,
                num_classes = 19,
                cuda = '0',
                use_gpu = True,
                save_model_path = './output_models',
                optimizer = 'adam',
                loss = 'crossentropy',
                #ours
                citySpaces_path = '',
                gta5_path = ''
                ):
        self.mode = mode
        self.backbone = backbone
        self.pretrain_path = pretrain_path
        self.use_conv_last = use_conv_last
        self.num_epochs = num_epochs
        self.epoch_start_i = epoch_start_i
        self.checkpoint_step = checkpoint_step
        self.validation_step = validation_step
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.cuda = cuda
        self.use_gpu = use_gpu
        self.save_model_path = save_model_path
        self.optimizer = optimizer
        self.loss = loss
        #ours
        self.citySpaces_path = citySpaces_path
        self.gta5_path = gta5_path