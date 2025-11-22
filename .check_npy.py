import os
import numpy as np
folder='data/save_dataset'
files=[f for f in os.listdir(folder) if f.endswith('.npy')]
print('npy files:', files)
if files:
    f=os.path.join(folder,files[0])
    arr=np.load(f,allow_pickle=True)
    print('loaded',files[0],'type',type(arr))
    try:
        print('len=',len(arr))
    except Exception as e:
        print('len error',e)
    try:
        preview=(arr[:3] if hasattr(arr,'__len__') else [arr])
        print('preview types:',[type(x) for x in preview])
        # If elements are arrays/lists, print shapes for first element
        elem=preview[0]
        if hasattr(elem,'shape'):
            print('elem shape:', getattr(elem,'shape'))
        else:
            print('elem type sample repr:', repr(elem)[:200])
    except Exception as e:
        print('preview error',e)
else:
    print('no files')
