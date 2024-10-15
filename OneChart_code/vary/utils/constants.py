CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
DEFAULT_NUMBER_TOKEN = "<Number>"


# CLIP_VIT_PATH = '/data/hypertext/xpkong/detvary/checkpoints/vit-large-patch14/'

CONVERSATION_DATA = {

    'cc665k': {
        'images': "/path_to/LLaVA1.5/images/",
        'annotations': "/path_to/LLaVA1.5/llava_v1_5_665k.json",
    },

    'render_chart_en': {
        'images': "/image_root_path/",
        'annotations': "/your_json.json",
    },

    'render_chart_zh': {
        'images': "/image_root_path/",
        'annotations': "/your_json.json",
    },


}
