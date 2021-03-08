"""Thin posterior"""

import h5py
import pickle as pkl


def thin_posterior(input_file, output_file, config):

    thin_idx = slice(config["start"], config["end"], config["by"])

    f = h5py.File(input_file, "r", rdcc_nbytes=1024 ** 3, rdcc_nslots=1e6)
    output_dict = dict(
        beta1=f["samples/beta1"][thin_idx],
        beta2=f["samples/beta2"][thin_idx],
        beta3=f["samples/beta3"][
            thin_idx,
        ],
        sigma=f["samples/sigma"][
            thin_idx,
        ],
        xi=f["samples/xi"][thin_idx],
        gamma0=f["samples/gamma0"][thin_idx],
        gamma1=f["samples/gamma1"][thin_idx],
        seir=f["samples/events"][thin_idx],
        init_state=f["initial_state"][:],
    )
    f.close()

    with open(output_file, "wb") as f:
        pkl.dump(
            output_dict,
            f,
        )


if __name__ == "__main__":

    import yaml
    import os
    from covid.cli_arg_parse import cli_args

    args = cli_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def join_and_expand(prefix, filename):
        return os.path.expandvars(os.path.join(prefix, filename))

    input_file = join_and_expand(
        config["output"]["results_dir"], config["output"]["posterior"]
    )
    output_file = join_and_expand(
        config["output"]["results_dir"], config["ThinPosterior"]["output_file"]
    )

    thin_idx = range(
        config["ThinPosterior"]["thin_start"],
        config["ThinPosterior"]["thin_end"],
        config["ThinPosterior"]["thin_by"],
    )
    thin_posterior(input_file, output_file, thin_idx)
