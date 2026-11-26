import copy

import pytest

torch = pytest.importorskip('torch')
from datagoblin.training_gym.mms import MechanisticMemory  # noqa: E402
from research.dg_gym.mod7_mms import ExposedGRUCell, keys_for  # noqa: E402


def test_gru_output_and_gradient_parity():
    torch.manual_seed(31)
    native=torch.nn.GRUCell(14,64)
    exposed=ExposedGRUCell(copy.deepcopy(native))
    x,h=torch.randn(5,14),torch.randn(5,64)
    expected=native(x,h)
    actual=exposed(x,h)
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for a,b in ((native.weight_ih,exposed.weight_ih),
                (native.weight_hh,exposed.recurrent.weight),
                (native.bias_ih,exposed.bias_ih),
                (native.bias_hh,exposed.recurrent.bias)):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)


def test_mms_observe_and_feedback_with_gru():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    cell=ExposedGRUCell(torch.nn.GRUCell(14,64))
    other=copy.deepcopy(cell)
    x,h=torch.randn(5,14),torch.randn(5,64)
    memory=MechanisticMemory(cell.recurrent,mib=.1,mode='observe')
    for step in range(2):
        cell.zero_grad()
        other.zero_grad()
        with memory.capture(torch.arange(5),step,1,2):
            cell(x,cell(x,h)).square().mean().backward()
        other(x,other(x,h)).square().mean().backward()
        assert torch.equal(cell.recurrent.weight.grad,other.recurrent.weight.grad)
    assert memory.bank.hits==5
    assert memory.admitted==0
    assert memory.bank.allocated_bytes <= memory.bank.budget_bytes


def test_length_and_program_keys():
    r=torch.tensor([0,0,1])
    ops=torch.tensor([[[1,0],[2,1]],[[1,0],[2,2]],[[1,0],[2,1]]])
    assert keys_for(r,ops).unique().numel()==3
    assert not torch.isin(keys_for(r,ops),keys_for(r,ops[:,:1])).any()
