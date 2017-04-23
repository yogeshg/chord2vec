
## Converting models to numpy format

* model files were converted to `npz`
``` python
sess = tf.InteractiveSession()
# run the auto_regressive code using this session for 0 epochs
gv = tf.global_variables()
np.savez('./data/model_199', wM1=gv[0].eval(), wM2=gv[1].eval(), wW=gv[2].eval(), bM2=gv[3].eval())

[(x.name, x.eval().shape, np.linalg.norm(x.eval())) for x in tf.global_variables()[:4]]

# [(u'Variable:0', (88, 1024), 297.28964),      # W
#  (u'Variable_1:0', (1024, 88), 299.65167),    # M
#  (u'Variable_2:0', (1024, 88), 296.36218),    # L
#  (u'Variable_3:0', (1024,), 31.940601)]       # M x + b
```

* files stored as npz can be read as

``` python
import numpy as np

npzf = np.load('data/model_199.npz')
                                    
npzf.files                          
['wW', 'wM2', 'wM1', 'bM2']         
                                    
for f in npzf.files:                
    print npzf[f].shape             

```

