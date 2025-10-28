"""Generate CRF confidence demo outputs and plot token/tag heatmap."""
import os
from typing import Iterable, List, Sequence

import numpy as np

from bert_base.train.confidence import (
    aggregate_span_confidence,
    compute_token_marginals,
)


def _build_sample() -> tuple:
    """Creates a small synthetic example for confidence visualization."""
    tokens: List[str] = ["厦", "门", "大", "学", "腾", "讯"]
    tags: List[str] = ["B-LOC", "I-LOC", "I-LOC", "O", "B-ORG", "I-ORG"]

    logits = np.array(
        [
            [3.0, 1.0, 0.1, 0.0, 0.0],
            [0.6, 3.1, 0.2, 0.1, 0.1],
            [0.4, 2.6, 0.3, 0.2, 0.2],
            [0.2, 0.3, 2.8, 0.1, 0.0],
            [0.3, 0.2, 0.1, 2.7, 0.5],
            [0.2, 0.3, 0.2, 0.4, 2.9],
        ],
        dtype=np.float64,
    )

    transition_params = np.array(
        [
            [0.5, 1.2, 0.1, 0.3, 0.2],
            [0.2, 0.9, 0.3, 0.1, 0.1],
            [0.4, 0.2, 0.8, 0.3, 0.2],
            [0.2, 0.1, 0.4, 0.6, 1.1],
            [0.2, 0.2, 0.5, 0.2, 0.9],
        ],
        dtype=np.float64,
    )

    label_to_id = {"B-LOC": 0, "I-LOC": 1, "O": 2, "B-ORG": 3, "I-ORG": 4}
    return tokens, tags, logits, transition_params, label_to_id


def main() -> None:
    tokens, tags, logits, transition_params, label_to_id = _build_sample()

    token_marginals = compute_token_marginals(
        logits[np.newaxis, ...], transition_params, [len(tokens)]
    )[0]
    span_confidences = aggregate_span_confidence(
        tags, token_marginals, label_to_id, tokens
    )

    sorted_labels = [
        label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]

    print("Token marginals (rows=tokens, cols=tags):")
    header = "{:<4s} ".format("") + " ".join(
        f"{label:>8s}" for label in sorted_labels
    )
    print(header)
    for token, probs in zip(tokens, token_marginals):
        formatted = " ".join(
            f"{probs[label_to_id[label]]:8.4f}" for label in sorted_labels
        )
        print(f"{token:<4s} {formatted}")

    print("\nSpan confidences:")
    for span in span_confidences:
        token_span = "".join(span.get("tokens", []))
        print(
            f"{token_span:<6s} ({span['type']}): min={span['min_prob']:.3f}, "
            f"mean={span['mean_prob']:.3f}, prod={span['prod_prob']:.3f}"
        )

    tag_list = sorted_labels
    output_dir = "pictures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confidence_demo.svg")
    _render_heatmap_svg(
        token_marginals,
        tokens,
        tag_list,
        [label_to_id[label] for label in tag_list],
        output_path,
    )
    print(f"\nSaved heatmap to {output_path}")


def _render_heatmap_svg(
    token_marginals: np.ndarray,
    tokens: Sequence[str],
    tag_list: Sequence[str],
    tag_indices: Iterable[int],
    output_path: str,
) -> None:
    cell_width = 80
    cell_height = 40
    margin_x = 80
    margin_y = 60
    width = margin_x + cell_width * len(tokens) + 20
    height = margin_y + cell_height * len(tag_list) + 60

    def prob_to_color(prob: float) -> str:
        prob = min(max(prob, 0.0), 1.0)
        start = np.array([240, 249, 232], dtype=np.float64)
        end = np.array([37, 83, 157], dtype=np.float64)
        rgb = start + (end - start) * prob
        r, g, b = [int(round(x)) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'",
        "     viewBox='0 0 {0} {1}'>".format(width, height),
        "  <rect x='0' y='0' width='{0}' height='{1}' fill='#ffffff' stroke='none'/>".format(
            width, height
        ),
        "  <text x='{0}' y='{1}' font-size='18' font-family='sans-serif'>CRF Token Marginals</text>".format(
            margin_x, 30
        ),
    ]

    for row, (tag, tag_index) in enumerate(zip(tag_list, tag_indices)):
        y = margin_y + row * cell_height
        lines.append(
            "  <text x='{0}' y='{1}' font-size='14' font-family='sans-serif'"
            " text-anchor='end' dominant-baseline='middle'>{2}</text>".format(
                margin_x - 10, y + cell_height / 2, tag
            )
        )
        for col, token in enumerate(tokens):
            x = margin_x + col * cell_width
            prob = float(token_marginals[col, tag_index])
            color = prob_to_color(prob)
            lines.append(
                "  <rect x='{0}' y='{1}' width='{2}' height='{3}' fill='{4}'".format(
                    x, y, cell_width, cell_height, color
                )
                + " stroke='#ffffff' stroke-width='1'/>"
            )
            lines.append(
                "  <text x='{0}' y='{1}' font-size='13' font-family='monospace'"
                " text-anchor='middle' dominant-baseline='middle'>{2:.2f}</text>".format(
                    x + cell_width / 2, y + cell_height / 2, prob
                )
            )

    for col, token in enumerate(tokens):
        x = margin_x + col * cell_width
        lines.append(
            "  <text x='{0}' y='{1}' font-size='14' font-family='sans-serif'"
            " text-anchor='middle'>{2}</text>".format(
                x + cell_width / 2, margin_y + cell_height * len(tag_list) + 30, token
            )
        )

    lines.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.write("\n".join(lines))


if __name__ == "__main__":
    main()
