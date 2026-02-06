#!/usr/bin/env python
"""Test script to verify the flow matching implementation changes.

This tests:
1. VectorFieldNetwork with reaction embedding
2. ConditionalFlowMatching with reaction conditioning
3. Scaffold split utility functions
4. NoveltyMetrics dataclass
"""

import sys
import numpy as np
import torch


def test_vector_field_network():
    """Test VectorFieldNetwork with reaction embedding."""
    print("Testing VectorFieldNetwork...")

    from synthemol.flows.flow_matching import VectorFieldNetwork

    # Create network with default parameters
    net = VectorFieldNetwork(
        input_dim=4096,
        cond_dim=2,
        hidden_dim=256,  # Smaller for testing
        num_layers=2,
        num_reactions=124,
        reaction_embed_dim=64,
    )

    batch_size = 4
    x = torch.randn(batch_size, 4096)
    t = torch.rand(batch_size)
    cond = torch.rand(batch_size, 2)
    reaction_idx = torch.randint(0, 124, (batch_size,))

    # Test forward pass without reaction
    v1 = net(x, t, cond, reaction_idx=None)
    assert v1.shape == (batch_size, 4096), f"Expected shape {(batch_size, 4096)}, got {v1.shape}"

    # Test forward pass with reaction
    v2 = net(x, t, cond, reaction_idx=reaction_idx)
    assert v2.shape == (batch_size, 4096), f"Expected shape {(batch_size, 4096)}, got {v2.shape}"

    # Outputs should be different when reaction is provided
    assert not torch.allclose(v1, v2), "Outputs should differ with/without reaction"

    print("  VectorFieldNetwork: PASSED")
    return True


def test_conditional_flow_matching():
    """Test ConditionalFlowMatching with reaction conditioning."""
    print("Testing ConditionalFlowMatching...")

    from synthemol.flows.flow_matching import ConditionalFlowMatching

    # Create model
    model = ConditionalFlowMatching(
        input_dim=4096,
        cond_dim=2,
        hidden_dim=256,
        num_layers=2,
        sigma=0.01,
        num_reactions=124,
        reaction_embed_dim=64,
    )

    batch_size = 8
    x_1 = torch.randn(batch_size, 4096)
    cond = torch.rand(batch_size, 2)
    reaction_idx = torch.randint(0, 124, (batch_size,))

    # Test forward pass without reaction
    loss1, metrics1 = model(x_1, cond, reaction_idx=None)
    assert loss1.shape == (), f"Loss should be scalar, got {loss1.shape}"
    assert "loss" in metrics1

    # Test forward pass with reaction
    loss2, metrics2 = model(x_1, cond, reaction_idx=reaction_idx)
    assert loss2.shape == (), f"Loss should be scalar, got {loss2.shape}"

    # Test sampling without reaction
    samples1 = model.sample(cond[:1], num_samples=2, num_steps=5)
    assert samples1.shape == (2, 4096), f"Expected shape (2, 4096), got {samples1.shape}"

    # Test sampling with reaction
    samples2 = model.sample(cond[:1], num_samples=2, num_steps=5, reaction_idx=reaction_idx[:1])
    assert samples2.shape == (2, 4096), f"Expected shape (2, 4096), got {samples2.shape}"

    print("  ConditionalFlowMatching: PASSED")
    return True


def test_scaffold_split_functions():
    """Test scaffold split utility functions."""
    print("Testing scaffold split functions...")

    # Import the functions from the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scaffold_split",
        "scripts/scaffold_split.py"
    )
    scaffold_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scaffold_module)

    get_scaffold = scaffold_module.get_scaffold
    scaffold_split = scaffold_module.scaffold_split

    # Test get_scaffold
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "c1ccccc1O",  # Phenol
        "c1ccccc1C",  # Toluene
    ]

    scaffolds = [get_scaffold(smi) for smi in test_smiles]
    assert len(scaffolds) == len(test_smiles)

    # Benzene, phenol, and toluene should have same scaffold
    assert scaffolds[2] == scaffolds[3] == scaffolds[4], "Benzene derivatives should have same scaffold"

    # Test scaffold_split
    train_idx, val_idx, test_idx = scaffold_split(
        test_smiles,
        train_ratio=0.6,
        val_ratio=0.2,
        random_state=42,
    )

    # Check no overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    assert len(train_set & val_set) == 0, "Train and val should not overlap"
    assert len(train_set & test_set) == 0, "Train and test should not overlap"
    assert len(val_set & test_set) == 0, "Val and test should not overlap"

    # Check all indices covered
    all_idx = train_set | val_set | test_set
    assert all_idx == set(range(len(test_smiles))), "All indices should be covered"

    print("  Scaffold split functions: PASSED")
    return True


def test_novelty_metrics():
    """Test NoveltyMetrics dataclass."""
    print("Testing NoveltyMetrics...")

    from synthemol.flows.synthesis_generator import NoveltyMetrics

    metrics = NoveltyMetrics(
        exact_novel=True,
        canonical_novel=True,
        max_tanimoto=0.75,
        structurally_novel=True,
        scaffold_novel=False,
        nearest_training_smiles="CCO",
    )

    assert metrics.exact_novel == True
    assert metrics.canonical_novel == True
    assert metrics.max_tanimoto == 0.75
    assert metrics.structurally_novel == True
    assert metrics.scaffold_novel == False
    assert metrics.nearest_training_smiles == "CCO"

    print("  NoveltyMetrics: PASSED")
    return True


def test_generated_molecule():
    """Test GeneratedMolecule with novelty_metrics field."""
    print("Testing GeneratedMolecule...")

    from synthemol.flows.synthesis_generator import GeneratedMolecule, NoveltyMetrics

    novelty = NoveltyMetrics(
        exact_novel=True,
        canonical_novel=True,
        max_tanimoto=0.65,
        structurally_novel=True,
        scaffold_novel=True,
    )

    mol = GeneratedMolecule(
        smiles="CCO",
        bb1_smiles="CC",
        bb2_smiles="O",
        reaction_id="22",
        target_activity=0.7,
        target_qed=0.8,
        actual_qed=0.75,
        is_novel=True,
        novelty_metrics=novelty,
    )

    assert mol.smiles == "CCO"
    assert mol.novelty_metrics is not None
    assert mol.novelty_metrics.max_tanimoto == 0.65

    print("  GeneratedMolecule: PASSED")
    return True


def test_training_script_imports():
    """Test that training script can be imported."""
    print("Testing training script imports...")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_flow_matching",
        "scripts/train_flow_matching.py"
    )
    train_module = importlib.util.module_from_spec(spec)

    # Check function signatures exist
    spec.loader.exec_module(train_module)

    assert hasattr(train_module, 'load_data')
    assert hasattr(train_module, 'create_dataloaders')
    assert hasattr(train_module, 'train_epoch')
    assert hasattr(train_module, 'validate')

    print("  Training script imports: PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Flow Matching Implementation Tests")
    print("=" * 60 + "\n")

    tests = [
        ("VectorFieldNetwork", test_vector_field_network),
        ("ConditionalFlowMatching", test_conditional_flow_matching),
        ("NoveltyMetrics", test_novelty_metrics),
        ("GeneratedMolecule", test_generated_molecule),
        ("Training Script", test_training_script_imports),
    ]

    # Scaffold split requires rdkit
    try:
        from rdkit import Chem
        tests.append(("Scaffold Split", test_scaffold_split_functions))
    except ImportError:
        print("Note: Skipping scaffold split tests (rdkit not installed)\n")

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
