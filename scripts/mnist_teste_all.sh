dataset=mnist
model=lenet3
st=irregular
prePGD=1636127344-0616112-c6.pt

pr=0.75
mix=0
lx=0.0002
ly=0.00005
save_name=testeall_prune_${pr}_${st}_mix_${mix}

run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml \
    -ep 3 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.0005 -bs 128 \
    -reopt adam --lr-scheduler default -ree 1 -relr 0.001 -relx ${lx} -rely ${ly} \
    -mf ${dataset}_${model}_${save_name}.pt \
    -lm ${prePGD} \
    --distill --distill_loss kl --distill_model_path ./assets/models/${prePGD} --distill_temp 30 --distill_alpha 10 \
    -adv adv --mix_ratio ${mix} --device cuda:0 \
    > logs/${dataset}/${save_name}.out 2>&1
