"""Known-answer M/C/J posterior semantics and shuffled-pair evaluator check."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiments/knot/scientific-closure-run2/synthetic_joint_closure.json"
SEED = 20261002
TRAIN_EVENTS = 32000
VALIDATION_EVENTS = 6000
STEPS = 1400
BATCH_SIZE = 256
LEARNING_RATE = 0.003
POSTERIOR_EVENTS = 2048
POSTERIOR_SAMPLES = 256


def generator_parameters(x):
    logits = 0.8 * x[..., 0] - 0.55 * x[..., 1]
    probability = torch.sigmoid(logits)
    mu0 = 0.65 * x[..., 0] - 0.25 * x[..., 1] - 0.8
    mu1 = 0.65 * x[..., 0] - 0.25 * x[..., 1] + 0.8
    return probability, mu0, mu1


def generate(x, rng):
    probability, mu0, mu1 = generator_parameters(x)
    a = torch.as_tensor(
        (rng.random(len(x)) < probability.cpu().numpy()).astype(np.float32)
    )
    sign = 2.0 * a - 1.0
    mu_n = torch.where(a == 1, mu1, mu0)
    n = mu_n + 0.45 * torch.as_tensor(rng.normal(size=len(x)), dtype=x.dtype)
    slope = 0.7 + 0.9 * a
    offset = 0.4 * x[:, 1] + 0.4 * sign
    p = slope * n + offset + 0.35 * torch.as_tensor(rng.normal(size=len(x)), dtype=x.dtype)
    return a, n, p


class MLP(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_width, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, output_width),
        )

    def forward(self, x):
        return self.net(x)


class ToyVariant(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.assignment = MLP(2, 1)
        if variant == "M":
            self.invisible = MLP(2, 2)
            self.parent = MLP(2, 2)
        elif variant == "C":
            self.invisible = MLP(3, 2)
            self.parent = MLP(4, 2)
        elif variant == "J":
            self.joint = MLP(3, 5)
        else:
            raise ValueError(variant)

    @staticmethod
    def gaussian_parameters(output):
        mean = output[..., 0]
        sigma = F.softplus(output[..., 1]) + 0.03
        return mean, sigma

    def loss(self, x, a, n, p):
        a_input = a[:, None]
        a_loss = F.binary_cross_entropy_with_logits(
            self.assignment(x)[:, 0], a
        )
        if self.variant == "M":
            n_mean, n_sigma = self.gaussian_parameters(self.invisible(x))
            p_mean, p_sigma = self.gaussian_parameters(self.parent(x))
            n_loss = normal_nll(n, n_mean, n_sigma)
            p_loss = normal_nll(p, p_mean, p_sigma)
            return a_loss + 0.5 * (n_loss + p_loss), {
                "assignment": a_loss,
                "invisible": n_loss,
                "parent": p_loss,
            }
        if self.variant == "C":
            n_mean, n_sigma = self.gaussian_parameters(
                self.invisible(torch.cat((x, a_input), dim=1))
            )
            p_mean, p_sigma = self.gaussian_parameters(
                self.parent(torch.cat((x, a_input, n[:, None]), dim=1))
            )
            n_loss = normal_nll(n, n_mean, n_sigma)
            p_loss = normal_nll(p, p_mean, p_sigma)
            return a_loss + 0.5 * (n_loss + p_loss), {
                "assignment": a_loss,
                "invisible": n_loss,
                "parent": p_loss,
            }
        output = self.joint(torch.cat((x, a_input), dim=1))
        joint_loss = bivariate_nll(n, p, output)
        return a_loss + 0.5 * joint_loss, {
            "assignment": a_loss,
            "joint": joint_loss,
        }

    @torch.no_grad()
    def sample(self, x, samples, rng):
        count = len(x)
        x_rep = x[:, None, :].expand(-1, samples, -1).reshape(-1, 2)
        probability = torch.sigmoid(self.assignment(x_rep)[:, 0]).reshape(count, samples)
        a = torch.as_tensor(
            (rng.random((count, samples)) < probability.cpu().numpy()).astype(np.float32)
        )
        a_rep = a.reshape(-1, 1)
        if self.variant == "M":
            n_mean, n_sigma = self.gaussian_parameters(self.invisible(x_rep))
            p_mean, p_sigma = self.gaussian_parameters(self.parent(x_rep))
            n = n_mean + n_sigma * torch.as_tensor(
                rng.normal(size=count * samples), dtype=x.dtype
            )
            p = p_mean + p_sigma * torch.as_tensor(
                rng.normal(size=count * samples), dtype=x.dtype
            )
        elif self.variant == "C":
            n_mean, n_sigma = self.gaussian_parameters(
                self.invisible(torch.cat((x_rep, a_rep), dim=1))
            )
            n = n_mean + n_sigma * torch.as_tensor(
                rng.normal(size=count * samples), dtype=x.dtype
            )
            p_mean, p_sigma = self.gaussian_parameters(
                self.parent(torch.cat((x_rep, a_rep, n[:, None]), dim=1))
            )
            p = p_mean + p_sigma * torch.as_tensor(
                rng.normal(size=count * samples), dtype=x.dtype
            )
        else:
            output = self.joint(torch.cat((x_rep, a_rep), dim=1))
            mu_n, mu_p, l11, l21, l22 = joint_parameters(output)
            eps1 = torch.as_tensor(rng.normal(size=count * samples), dtype=x.dtype)
            eps2 = torch.as_tensor(rng.normal(size=count * samples), dtype=x.dtype)
            n = mu_n + l11 * eps1
            p = mu_p + l21 * eps1 + l22 * eps2
        return (
            a.numpy().astype(np.float64),
            n.reshape(count, samples).numpy().astype(np.float64),
            p.reshape(count, samples).numpy().astype(np.float64),
        )


def normal_nll(target, mean, sigma):
    return (0.5 * ((target - mean) / sigma).square() + sigma.log()).mean()


def joint_parameters(output):
    mu_n = output[..., 0]
    mu_p = output[..., 1]
    l11 = F.softplus(output[..., 2]) + 0.03
    l21 = output[..., 3]
    l22 = F.softplus(output[..., 4]) + 0.03
    return mu_n, mu_p, l11, l21, l22


def bivariate_nll(n, p, output):
    mu_n, mu_p, l11, l21, l22 = joint_parameters(output)
    z1 = (n - mu_n) / l11
    z2 = (p - mu_p - l21 * z1) / l22
    return (0.5 * (z1.square() + z2.square()) + l11.log() + l22.log()).mean()


def evaluate_loss(model, x, a, n, p):
    with torch.no_grad():
        total, components = model.loss(x, a, n, p)
    return {
        "total": float(total),
        "components": {key: float(value) for key, value in components.items()},
    }


def sample_truth_posterior(x, samples, rng):
    rows = len(x)
    x_rep = x[:, None, :].expand(-1, samples, -1).reshape(-1, 2)
    a, n, p = generate(x_rep, rng)
    return (
        a.reshape(rows, samples).numpy().astype(np.float64),
        n.reshape(rows, samples).numpy().astype(np.float64),
        p.reshape(rows, samples).numpy().astype(np.float64),
    )


def joint_metric(n, p):
    n0 = n - n.mean(axis=1, keepdims=True)
    p0 = p - p.mean(axis=1, keepdims=True)
    covariance = np.mean(n0 * p0, axis=1)
    n_var = np.mean(n0 * n0, axis=1)
    p_var = np.mean(p0 * p0, axis=1)
    correlation = covariance / np.sqrt(np.maximum(n_var * p_var, 1.0e-12))
    return {
        "mean_conditional_covariance": float(covariance.mean()),
        "mean_conditional_correlation": float(correlation.mean()),
    }


def marginal_metric(n, p):
    return {
        "N": {"mean": float(n.mean()), "variance": float(n.var())},
        "P": {"mean": float(p.mean()), "variance": float(p.var())},
    }


def oracle_metrics(model, x, a_truth, n_truth, p_truth, rng):
    a_pred, n_pred, _ = model.sample(x, 1, rng)
    a_pred = torch.as_tensor(a_pred[:, 0], dtype=torch.float32)
    n_pred = torch.as_tensor(n_pred[:, 0], dtype=torch.float32)
    x = x.to(torch.float32)
    a_truth = a_truth.to(torch.float32)
    n_truth = n_truth.to(torch.float32)
    p_truth = p_truth.to(torch.float32)
    with torch.no_grad():
        n_true_mean, n_true_sigma = model.gaussian_parameters(
            model.invisible(torch.cat((x, a_truth[:, None]), dim=1))
        )
        n_pred_mean, n_pred_sigma = model.gaussian_parameters(
            model.invisible(torch.cat((x, a_pred[:, None]), dim=1))
        )
        p_true_mean, p_true_sigma = model.gaussian_parameters(
            model.parent(torch.cat((x, a_truth[:, None], n_truth[:, None]), dim=1))
        )
        p_pred_mean, p_pred_sigma = model.gaussian_parameters(
            model.parent(torch.cat((x, a_pred[:, None], n_pred[:, None]), dim=1))
        )
    return {
        "N_given_A_truth_nll": float(normal_nll(n_truth, n_true_mean, n_true_sigma)),
        "N_given_A_pred_nll": float(normal_nll(n_truth, n_pred_mean, n_pred_sigma)),
        "P_given_A_truth_N_truth_nll": float(normal_nll(p_truth, p_true_mean, p_true_sigma)),
        "P_given_A_pred_N_pred_nll": float(normal_nll(p_truth, p_pred_mean, p_pred_sigma)),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    x_train = torch.as_tensor(rng.normal(size=(TRAIN_EVENTS, 2)), dtype=torch.float32)
    a_train, n_train, p_train = generate(x_train, rng)
    x_valid = torch.as_tensor(rng.normal(size=(VALIDATION_EVENTS, 2)), dtype=torch.float32)
    a_valid, n_valid, p_valid = generate(x_valid, rng)
    batch_rng = np.random.default_rng(SEED + 1)
    batches = batch_rng.integers(0, TRAIN_EVENTS, size=(STEPS, BATCH_SIZE))
    torch.set_num_threads(1)
    model_results = {}
    predictions = {}
    heldout_probabilities = {}

    for variant in ("M", "C", "J"):
        torch.manual_seed(SEED + 10)
        model = ToyVariant(variant)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()
        for step in range(STEPS):
            idx = torch.as_tensor(batches[step], dtype=torch.long)
            loss = model.loss(
                x_train[idx], a_train[idx], n_train[idx], p_train[idx]
            )[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant}: nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        valid_loss = evaluate_loss(model, x_valid, a_valid, n_valid, p_valid)
        sample_events = x_valid[:POSTERIOR_EVENTS]
        a_samples, n_samples, p_samples = model.sample(
            sample_events, POSTERIOR_SAMPLES, np.random.default_rng(SEED + 3)
        )
        predictions[variant] = (a_samples, n_samples, p_samples)
        with torch.no_grad():
            heldout_probabilities[variant] = torch.sigmoid(
                model.assignment(x_valid)[:, 0]
            ).numpy()
        model_results[variant] = {
            "factorization": {
                "M": "q(A|x) q(N|x) q(P|x)",
                "C": "q(A|x) q(N|A,x) q(P|A,N,x)",
                "J": "q(A|x) q(N,P|A,x)",
            }[variant],
            "training": {
                "conditioning": "truth",
                "events": TRAIN_EVENTS,
                "steps": STEPS,
                "batch_size": BATCH_SIZE,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "final_train_loss": float(loss.detach()),
                "validation_loss": valid_loss,
            },
            "posterior_samples": POSTERIOR_SAMPLES,
            "N_P_coherent": joint_metric(n_samples, p_samples),
        }
        if variant == "C":
            model_results[variant]["oracle"] = oracle_metrics(
                model, x_valid, a_valid, n_valid, p_valid, np.random.default_rng(SEED + 4)
            )
        del model, optimizer

    sample_events = x_valid[:POSTERIOR_EVENTS]
    _, n_truth, p_truth = sample_truth_posterior(
        sample_events, POSTERIOR_SAMPLES, np.random.default_rng(SEED + 5)
    )
    truth_marginals = marginal_metric(n_truth, p_truth)
    truth_joint = joint_metric(n_truth, p_truth)
    shuffled_metrics = {}
    permutation_rng = np.random.default_rng(SEED + 6)
    marginal_delta = 0.0
    for variant, (_a, n_samples, p_samples) in predictions.items():
        permutation = np.argsort(
            permutation_rng.random(p_samples.shape), axis=1, kind="stable"
        )
        p_shuffled = np.take_along_axis(p_samples, permutation, axis=1)
        coherent_marginal = marginal_metric(n_samples, p_samples)
        shuffled_marginal = marginal_metric(n_samples, p_shuffled)
        delta = max(
            abs(coherent_marginal[source][field] - shuffled_marginal[source][field])
            for source in ("N", "P")
            for field in ("mean", "variance")
        )
        marginal_delta = max(marginal_delta, delta)
        coherent_joint = joint_metric(n_samples, p_samples)
        shuffled_joint = joint_metric(n_samples, p_shuffled)
        shuffled_metrics[variant] = {
            "marginals_coherent": coherent_marginal,
            "marginals_shuffled": shuffled_marginal,
            "marginal_max_absolute_change": delta,
            "joint_coherent": coherent_joint,
            "joint_shuffled": shuffled_joint,
            "coherent_covariance_error": abs(
                coherent_joint["mean_conditional_covariance"]
                - truth_joint["mean_conditional_covariance"]
            ),
            "shuffled_covariance_error": abs(
                shuffled_joint["mean_conditional_covariance"]
                - truth_joint["mean_conditional_covariance"]
            ),
        }

    dependence = {
        "ambiguous_A_probability_min": float(
            heldout_probabilities["C"].min()
        ),
        "ambiguous_A_probability_max": float(
            heldout_probabilities["C"].max()
        ),
        "A_truth_entropy_bits": float(
            np.mean(
                -(
                    heldout_probabilities["C"] * np.log2(heldout_probabilities["C"])
                    + (1.0 - heldout_probabilities["C"])
                    * np.log2(1.0 - heldout_probabilities["C"])
                )
            )
        ),
        "N_conditional_mean_separation_between_A_values": 1.6,
        "P_conditional_slope_N_by_A": {"A0": 0.7, "A1": 1.6},
        "truth_generator_mean_conditional_N_P_covariance": truth_joint[
            "mean_conditional_covariance"
        ],
        "truth_generator_mean_conditional_N_P_correlation": truth_joint[
            "mean_conditional_correlation"
        ],
    }
    evaluator_valid = (
        marginal_delta < 1.0e-10
        and all(
            shuffled_metrics[variant]["shuffled_covariance_error"]
            > shuffled_metrics[variant]["coherent_covariance_error"]
            for variant in ("C", "J")
        )
        and all(
            abs(shuffled_metrics[variant]["joint_coherent"]["mean_conditional_covariance"])
            > abs(shuffled_metrics[variant]["joint_shuffled"]["mean_conditional_covariance"])
            for variant in ("C", "J")
        )
    )
    result = {
        "toy": {
            "generator": "x~N(0,I2); A~Bernoulli(sigmoid(0.8*x0-0.55*x1)); N=0.65*x0-0.25*x1+0.8*(2A-1)+epsN; P=(0.7+0.9A)*N+0.4*x1+0.4*(2A-1)+epsP",
            "sigma_N": 0.45,
            "sigma_P": 0.35,
            "train_events": TRAIN_EVENTS,
            "validation_events": VALIDATION_EVENTS,
            "sealed_test_used": False,
        },
        "conditioning_checks": dependence,
        "M_C_J": model_results,
        "joint_evaluator": {
            "permutation_seed": SEED + 6,
            "permutation": "within each event, permute P across posterior-sample index",
            "truth_marginals": truth_marginals,
            "truth_joint": truth_joint,
            "results": shuffled_metrics,
            "maximum_marginal_change_under_shuffle": marginal_delta,
            "KNOT_JOINT_EVALUATOR_VALID": evaluator_valid,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "conditioning_checks": dependence,
        "M_C_J": {
            key: {
                "validation_loss": value["training"]["validation_loss"],
                "N_P_coherent": value["N_P_coherent"],
                "oracle": value.get("oracle"),
            }
            for key, value in model_results.items()
        },
        "joint_evaluator": result["joint_evaluator"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
