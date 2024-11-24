import tqdm

from utils.geometry_utils import (
    convert_conventions,
    generate_random_transform,
)
import magnum as mn
import numpy as np


def assert_vectors(vec_hab, vec_std, atol=1e-4):
    try:
        assert np.allclose(vec_hab[0], -vec_std[1], atol=atol)
        assert np.allclose(vec_hab[1], vec_std[2], atol=atol)
        assert np.allclose(vec_hab[2], -vec_std[0], atol=atol)
    except AssertionError:
        raise AssertionError(
            "Vectors do not match between habitat and standard conventions. "
            "Check the conversion functions.\n"
            f" vec_hab: {vec_hab}\n"
            f" vec_std: {vec_std}\n"
            f" vec_hab[0], -vec_std[1]: {vec_hab[0], -vec_std[1]}\n"
            f" vec_hab[1], vec_std[2]: {vec_hab[1], vec_std[2]}\n"
            f" vec_hab[2], -vec_std[0]: {vec_hab[2], -vec_std[0]}\n"
        )


def check_transforms(tf_hab, tf_std):
    assert_vectors(tf_hab.translation, tf_std.translation)

    local_vector = mn.Vector3(np.random.rand(3))

    global_vector_hab = tf_hab.transform_point(local_vector)
    global_vector_std = tf_std.transform_point(local_vector)

    assert_vectors(global_vector_hab, global_vector_std)

    global_vector_hab_recon = convert_conventions(global_vector_std, reverse=True)
    global_vector_hab_np = np.array(global_vector_hab)
    global_vector_hab_recon_np = np.array(global_vector_hab_recon)

    assert np.allclose(global_vector_hab_np, global_vector_hab_recon_np, atol=1e-6)


def test1():
    for _ in tqdm.trange(1000):
        tf_hab = mn.Matrix4(generate_random_transform())
        tf_std = convert_conventions(tf_hab)
        check_transforms(tf_hab, tf_std)
    print("Test 1 (random transforms with transform_point) passed!")


def test2():
    for _ in tqdm.trange(1000):
        # Confirm chaining of transformations
        global_T_A = mn.Matrix4(generate_random_transform())
        B_T_A = mn.Matrix4(generate_random_transform())
        C_T_B = mn.Matrix4(generate_random_transform())
        D_T_C = mn.Matrix4(generate_random_transform())

        global_T_D_std = (  #
            global_T_A  #
            @ B_T_A.inverted()  #
            @ C_T_B.inverted()  #
            @ D_T_C.inverted()  #
        )

        global_T_A_hab = convert_conventions(global_T_A, reverse=True)
        B_T_A_hab = convert_conventions(B_T_A, reverse=True)
        C_T_B_hab = convert_conventions(C_T_B, reverse=True)
        D_T_C_hab = convert_conventions(D_T_C, reverse=True)
        global_T_D_hab = (  #
            global_T_A_hab  #
            @ B_T_A.inverted()  #
            @ C_T_B.inverted()  #
            @ D_T_C.inverted()  #
        )
        check_transforms(global_T_A_hab, global_T_A)
        check_transforms(B_T_A_hab, B_T_A)
        check_transforms(C_T_B_hab, C_T_B)
        check_transforms(D_T_C_hab, D_T_C)
        check_transforms(global_T_D_hab, global_T_D_std)
    print("Test 2 (chaining) passed!")


def test3():
    vecs = np.eye(3)[:2].tolist()
    vecs += (-np.eye(3))[:2].tolist()
    for idx, vec_std in enumerate(vecs):
        tf_std = mn.Matrix4().look_at(
            eye=mn.Vector3(vec_std),
            target=mn.Vector3(0, 0, 0),
            up=mn.Vector3(0, 0, 1),
        )
        vec_hab = convert_conventions(mn.Vector3(vec_std), reverse=True)
        tf_hab = mn.Matrix4().look_at(
            eye=mn.Vector3(vec_hab),
            target=mn.Vector3(0, 0, 0),
            up=mn.Vector3(0, 1, 0),
        )
        check_transforms(tf_hab, tf_std)
    print("Test 3 (look_at simple) passed!")


def test4():
    for _ in tqdm.trange(1000):
        eye_std = mn.Vector3(np.random.rand(3))
        target_std = mn.Vector3(np.random.rand(3))
        tf_std = mn.Matrix4().look_at(
            eye=eye_std,
            target=target_std,
            up=mn.Vector3(0, 0, 1),
        )
        eye_hab = convert_conventions(eye_std, reverse=True)
        target_hab = convert_conventions(target_std, reverse=True)
        tf_hab = mn.Matrix4().look_at(
            eye=eye_hab,
            target=target_hab,
            up=mn.Vector3(0, 1, 0),
        )

        tf_hab_np = np.array(tf_hab)
        tf_hab_reconstructed = convert_conventions(tf_std, reverse=True)
        tf_hab_reconstructed_np = np.array(tf_hab_reconstructed)

        tf_std_np = np.array(tf_std)
        tf_std_reconstructed = convert_conventions(tf_hab_reconstructed)
        tf_std_reconstructed_np = np.array(tf_std_reconstructed)
        assert np.allclose(tf_hab_np, tf_hab_reconstructed_np, atol=1e-6), (
            f"tf_hab_np: {tf_hab_np}\n"
            f"tf_hab_reconstructed_np: {tf_hab_reconstructed_np}"
        )
        assert np.allclose(tf_std_np, tf_std_reconstructed_np, atol=1e-6), (
            f"tf_std_np: {tf_std_np}\n"
            f"tf_std_reconstructed_np: {tf_std_reconstructed_np}"
        )
    print("Test 4 (look_at) passed!")


def test5():
    local_frame_std = mn.Matrix4().from_(
        mn.Matrix3x3(),
        mn.Vector3(1, 0, 0),
    )
    local_frame_hab = convert_conventions(local_frame_std, reverse=True)

    local_offset = mn.Vector3(1, 0, 0)
    local_offset = local_offset

    global_std = local_frame_std.transform_point(local_offset)
    global_hab = local_frame_hab.transform_point(local_offset)

    global_std_reconstructed = convert_conventions(global_hab)
    assert np.allclose(global_std, global_std_reconstructed, atol=1e-6)
    print("Test 5 (transform_point) passed!")


test1()
test2()
test3()
test4()
test5()
