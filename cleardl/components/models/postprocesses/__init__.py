from .nms import SingleLabelNMS, MultiLabelNMS

POSTPROCESSES = {
    'SingleLabelNMS': SingleLabelNMS,
    'MultiLabelNMS': MultiLabelNMS
}


def build_postprocess(postprocess: dict):
    return POSTPROCESSES[postprocess.pop('type')](**postprocess)
