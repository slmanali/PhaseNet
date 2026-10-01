import json, random
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from net.complex_nn.complex_layers import (ComplexConv2d, SeparateRealConv2d,
    UnrestrictedRealConv2d)
from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe
from tools.aggregate_seeded_results import aggregate, paired
from tools.benchmark_interpolation import count_conv_macs
from utils.reproducibility import (configure_reproducibility, load_checkpoint,
    make_data_generator, save_training_checkpoint, seed_worker)
from utils.inference import prepare_complex_channel
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from train_complex_safe_baseline import get_complex_input


def tiny_run(seed, width=4, checkpoint=None, resume=None, stop=2):
    configure_reproducibility(seed,"strict"); data_gen=make_data_generator(seed)
    model=nn.Sequential(nn.Linear(2,width),nn.ReLU(),nn.Linear(width,1)); initial=model[0].weight.detach().clone()
    opt=torch.optim.SGD(model.parameters(),.02); sched=torch.optim.lr_scheduler.StepLR(opt,1,.9)
    start=0; step=0
    if resume: state=load_checkpoint(resume,model,opt,sched,data_gen); start=state["epoch"]; step=state["global_step"]
    order=[]; losses=[]; ds=TensorDataset(torch.arange(16,dtype=torch.float32).view(8,2),torch.arange(8,dtype=torch.float32).view(8,1))
    for epoch in range(start,stop):
        loader=DataLoader(ds,batch_size=2,shuffle=True,generator=data_gen,worker_init_fn=seed_worker)
        for x,y in loader:
            order.extend((x[:,0]/2).long().tolist()); opt.zero_grad(); loss=((model(x)-y)**2).mean(); loss.backward(); opt.step(); losses.append(loss.item()); step+=1
        sched.step()
        if checkpoint is not None:
            save_training_checkpoint(checkpoint,model,opt,sched,epoch+1,step,{"seed":seed},data_gen)
    return model,initial,order,losses


def test_short_training_seed_and_order_reproducible():
    a,ai,ao,al=tiny_run(7,width=3); b,bi,bo,bl=tiny_run(7,width=5)
    c,ci,co,cl=tiny_run(7,width=3); _,di,_,_=tiny_run(8,width=3)
    assert ao==bo and ao==co and al==cl
    assert torch.equal(ai,ci) and not torch.equal(ai,di)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),c.parameters()))


def test_epoch_boundary_resume(tmp_path):
    checkpoint=tmp_path/'resume.pt'; full,*_=tiny_run(13)
    tiny_run(13,checkpoint=checkpoint,stop=1)
    resumed,*_=tiny_run(13,resume=checkpoint,stop=2)
    assert all(torch.equal(x,y) for x,y in zip(full.parameters(),resumed.parameters()))


@pytest.mark.parametrize("cls,factor",[(ComplexConv2d,2),(SeparateRealConv2d,2),(UnrestrictedRealConv2d,4)])
def test_convolution_shapes_gradients_and_weight_counts(cls,factor):
    layer=cls(2,3,3,padding=1,bias=True); r=torch.randn(2,2,5,5,requires_grad=True); i=torch.randn_like(r,requires_grad=True)
    yr,yi=layer(r,i); (yr.square().mean()+yi.square().mean()).backward()
    assert yr.shape==yi.shape==(2,3,5,5); assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
    weights=sum(p.numel() for n,p in layer.named_parameters() if "weight" in n)
    assert weights==factor*2*3*3*3


def test_block_matrix_initialization_matches_complex():
    torch.manual_seed(2); source=ComplexConv2d(2,3,3,padding=1); target=UnrestrictedRealConv2d(2,3,3,padding=1); target.initialize_from_complex(source)
    r=torch.randn(2,2,5,5); i=torch.randn_like(r)
    actual=target(r,i); expected=source(r,i)
    assert all(torch.allclose(a,b,atol=2e-6) for a,b in zip(actual,expected))


def test_ablation_keeps_non_convolution_state_aligned():
    torch.manual_seed(4); reference=ComplexPhaseNetSafe(4); target=ComplexPhaseNetSafe(4,convolution="unrestricted_real"); target.initialize_ablation_from(reference)
    ref=dict(reference.named_modules()); dst=dict(target.named_modules())
    assert torch.equal(reference.alpha,target.alpha)
    for name,module in dst.items():
        if "bn" in name and hasattr(module,"state_dict"):
            for k,v in module.state_dict().items(): assert torch.equal(v,ref[name].state_dict()[k])


def test_call_based_mac_count_catches_shared_module_calls():
    conv=nn.Conv2d(1,1,2,bias=False); x=torch.ones(1,1,3,3)
    class Shared(nn.Module):
        def __init__(self): super().__init__(); self.conv=conv
        def forward(self,z): return self.conv(z)+self.conv(z)
    model=Shared(); assert count_conv_macs(model,lambda:model(x))==2*2*2*2*2


def test_endpoint_preparation_matches_evaluator_and_ignores_middle():
    pyramid=SCFpyr_PyTorch(height=2,nbands=4,scale_factor=2,device=torch.device("cpu"))
    start=torch.rand(1,1,24,24); end=torch.rand_like(start)
    endpoint_r,endpoint_i,hp=prepare_complex_channel(start,end,pyramid)
    prepared=[]
    for middle in (torch.zeros_like(start),torch.rand_like(start)):
        coeff=pyramid.build(torch.cat((start,middle,end),dim=0),pyr_type=1)
        train_r,train_i,_,_,hp_start,hp_end=get_complex_input([coeff])
        prepared.append((train_r,train_i,.5*(hp_start+hp_end)))
    for train_r,train_i,legacy_hp in prepared:
        assert all(torch.allclose(a,b,atol=2e-5) for a,b in zip(endpoint_r,train_r))
        assert all(torch.allclose(a,b,atol=2e-5) for a,b in zip(endpoint_i,train_i))
        assert torch.allclose(hp,legacy_hp,atol=2e-5)


def test_aggregator_sample_std_incomplete_and_pairing():
    rows=[{"variant":"a","dataset":"d","seed":11,"status":"completed","psnr":1,"ssim":2,"lpips":3,"pce":4},
          {"variant":"a","dataset":"d","seed":22,"status":"completed","psnr":3,"ssim":4,"lpips":5,"pce":6},
          {"variant":"a","dataset":"d","seed":33,"status":"failed"},
          {"variant":"b","dataset":"d","seed":11,"status":"completed","psnr":0,"ssim":1,"lpips":2,"pce":3}]
    result=aggregate(rows,[11,22,33]); a=result["aggregates"][0]
    assert a["psnr_mean"]==2 and a["psnr_sample_std"]==pytest.approx(2**.5) and a["missing_seeds"]==[33]
    assert len(result["failed_or_incomplete"])==1 and paired(rows,"a","b")[0]["psnr_difference"]==1
