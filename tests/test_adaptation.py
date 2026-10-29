import unittest
import torch
from bloodcells.adaptation import LoRALinear, EpisodicProjection, proto_logits


class AdaptationComponents(unittest.TestCase):
    def test_lora_initial_equivalence_and_only_adapters_train(self):
        torch.manual_seed(7)
        base=torch.nn.Linear(12,8)
        adapted=LoRALinear(base,rank=2,alpha=4)
        x=torch.randn(4,12)
        torch.testing.assert_close(base(x),adapted(x))
        before=base.weight.detach().clone()
        opt=torch.optim.Adam([p for p in adapted.parameters() if p.requires_grad],lr=.1)
        opt.zero_grad(); adapted(x).square().mean().backward(); opt.step()
        self.assertFalse(base.weight.requires_grad)
        torch.testing.assert_close(base.weight,before)
        self.assertGreater(float(adapted.b.abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in adapted.parameters() if p.requires_grad),40)

    def test_episodic_projection_has_trainable_metric(self):
        model=EpisodicProjection(dim=4,hidden=3)
        x=torch.tensor([[1.,0,0,0],[0,1.,0,0]])
        y=torch.tensor([0,1])
        torch.testing.assert_close(model(x),x)
        opt=torch.optim.Adam(model.parameters(),lr=.1)
        before=model.net[-1].weight.detach().clone()
        opt.zero_grad()
        logits=proto_logits(model(x),y,model(x.flip(0)))
        torch.nn.functional.cross_entropy(logits,y).backward(); opt.step()
        self.assertFalse(torch.equal(before,model.net[-1].weight))

    def test_prototypes_classify_nearest_center(self):
        support=torch.tensor([[0.,0.],[.1,0.],[2.,2.],[2.,2.1]])
        labels=torch.tensor([0,0,1,1])
        query=torch.tensor([[.05,0.],[2.,2.05]])
        self.assertEqual(proto_logits(support,labels,query).argmax(1).tolist(),[0,1])


if __name__=='__main__': unittest.main()
