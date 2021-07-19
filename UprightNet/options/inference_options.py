from .base_options import BaseOptions

class InferenceOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--img_path', type=str, required=True, help='Image for inference')

        self.isTrain = True
