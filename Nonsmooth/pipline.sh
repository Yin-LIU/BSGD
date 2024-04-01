#!/bin/bash


# Required datafiles data.npy and index_rebalanced.pt from https//github.com/daniellevy/fast-dro/tree/main and https//www.dropbox.com/s/e0puwp86vyh4dkg/digits_data.zip?dl=0
# 1. Generate the data following Levy20
#python CreatDigitData.py
# 2. Evaluate bias and var v.s. inner batchsize
#python EvaluateBiasVar.py

# # -------- proxSGD -----------
# python main.py --seed 6 --algorithm proxSGD --stepsize 5e-3 --batch_allocation 2000000 --batch_size 10000 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10

# python main.py --seed 6 --algorithm proxSGD --stepsize 5e-4 --batch_allocation 2000000 --batch_size 1000 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10

# python main.py --seed 6 --algorithm proxSGD --stepsize 5e-5 --batch_allocation 2000000 --batch_size 100 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10

# python main.py --seed 6 --algorithm proxSGD --stepsize 5e-6 --batch_allocation 2000000 --batch_size 10 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10

# # python main.py --seed 6 --algorithm proxSGD --stepsize 5e-3 --batch_allocation 2000000 --batch_size 5000 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10

# # python main.py --seed 6 --algorithm proxSGD --stepsize 2e-4 --batch_allocation 2000000 --batch_size 500 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10



# # -------- proxABG -----------
# python main.py  --algorithm proxABG --stepsize 5e-3  --batch_size 10 --ABG_multiplier 1e4 --ABG_exponent -1 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# python main.py  --algorithm proxABG --stepsize 5e-3  --batch_size 10 --ABG_multiplier 1e3 --ABG_exponent -3 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# python main.py  --algorithm proxABG --stepsize 5e-4  --batch_size 10 --ABG_multiplier 1e2 --ABG_exponent -5 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000


# # ---------- multilevel ------------
# python main.py  --algorithm multilevel --stepsize 2e-4  --batch_size 100 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# # ---------- dual ------------
# python main.py --algorithm dual --stepsize 2e-3 --stepsize_eta 1e-3 --batch_size 1000 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# # ---------- primaldual ----------
# python main.py --algorithm primaldual --stepsize 1e-2 --stepsize_dual 3e-6 --batch_size 1000 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# ------------ ABG-STORM ------------
# python main.py --algorithm ABG_STORM --stepsize 1e-2 --batch_size 10 --ABG_multiplier 1e2 --ABG_exponent -5 --STORM_beta 0.5 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# python main.py --algorithm ABG_STORM --stepsize 5e-3 --batch_size 10 --ABG_multiplier 1e3 --ABG_exponent -5 --STORM_beta 0.5 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

# ------------ Multistage-STORM ------------
python main.py --algorithm Multistage_STORM --stepsize 2e-2 --batch_size 10  --Multi_STORM_loop 100 --STORM_beta 0.2 --regularizer l1 --regularizer_strength 0.0001 --num_trail 10 --seed 6 --batch_allocation 2000000

