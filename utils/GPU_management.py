#import configs.local_config as local_config
import os
import GPUtil
import time

def Assign_GPU(max_GPUs=1,**kwargs):
    #if 'GPU2USE' in local_config.__dir__():
    #    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_config.GPU2USE)
    #    os.environ["CUDA_AVAILABLE_DEVICES"] = str(local_config.GPU2USE)
    #    return [local_config.GPU2USE]
    excluded_IDs = []
    def getAvailable():
        return GPUtil.getAvailable(order='memory',excludeID=excluded_IDs,limit=max_GPUs if max_GPUs is not None else 100,**kwargs)
    GPU_2_use = getAvailable()
    if len(GPU_2_use)==0:
        # GPU_2_use = [0]
        print('No available GPUs. waiting...')
        while len(GPU_2_use)==0:
            time.sleep(5)
            GPU_2_use = getAvailable()
    assert len(GPU_2_use)>0,'No available GPUs...'
    if max_GPUs is not None:
        print('Using GPU #%d'%(GPU_2_use[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(GPU_2_use[0]) # Limit to 1 GPU when using an interactive session
        return [GPU_2_use[0]]
    else:
        return GPU_2_use
