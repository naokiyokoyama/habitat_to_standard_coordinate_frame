from utils.geometry_utils import convert_conventions, generate_random_transform
import magnum as mn
import numpy as np
import tqdm


def check_transforms(tf_hab, tf_std):
    local_vector_hab = mn.Vector3(np.random.rand(3))
    local_vector_std = convert_conventions(local_vector_hab)

    assert local_vector_hab[0] == -local_vector_std[1]
    assert local_vector_hab[1] == local_vector_std[2]
    assert local_vector_hab[2] == -local_vector_std[0]

    global_vector_hab = tf_hab.transform_vector(local_vector_hab)
    global_vector_std = tf_std.transform_vector(local_vector_std)

    assert np.allclose(global_vector_hab[0], -global_vector_std[1], atol=1e-6)
    assert np.allclose(global_vector_hab[1], global_vector_std[2], atol=1e-6)
    assert np.allclose(global_vector_hab[2], -global_vector_std[0], atol=1e-6)

    global_vector_hab_recon = convert_conventions(global_vector_std, reverse=True)
    global_vector_hab_np = np.array(global_vector_hab)
    global_vector_hab_recon_np = np.array(global_vector_hab_recon)

    assert np.allclose(global_vector_hab_np, global_vector_hab_recon_np, atol=1e-6)


for _ in tqdm.trange(5000):
    tf_hab = mn.Matrix4(generate_random_transform())
    tf_std = convert_conventions(tf_hab)
    check_transforms(tf_hab, tf_std)

    # Confirm chaining of transformations
    tf_list_hab = [mn.Matrix4(generate_random_transform()) for _ in range(10)]
    tf_list_std = [convert_conventions(tf) for tf in tf_list_hab]

    tf_chain_hab = tf_list_hab[0]
    for tf in tf_list_hab[1:]:
        tf_chain_hab = tf_chain_hab @ tf
    tf_chain_std = tf_list_std[0]
    for tf in tf_list_std[1:]:
        tf_chain_std = tf_chain_std @ tf

    check_transforms(tf_chain_hab, tf_chain_std)
