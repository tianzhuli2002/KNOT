# Topology format

A topology file contains one `topology` mapping. Each decay node contains named roles.
There are exactly three role forms:

```yaml
topology:
  Whad:
    q:
      - [JET, 3]
      - [JET, 4]
  thad:
    W: Whad
    b: [JET, 2]
  tlep:
    b: [JET, 1]
    lepton: [LEPTON, 1]
    nu: [NEUTRINO, 0]
```

- `[SOURCE, TRUTH_INDEX]` is one leaf.
- A list of leaves under the same named role is one exchangeable group. Thus
  `Whad.q[0] <-> Whad.q[1]`; roles with different names are never exchangeable.
- A string is a reference to another decay node. References infer the hierarchy;
  declaration order has no meaning.
- `NEUTRINO` is invisible truth. MET is detector context and is not a decay leaf.

The compiler rejects unknown keys and sources, malformed leaves, duplicate truth
ownership, undefined references and cycles. It derives roots, descendants, visible
support, invisible completion, candidate ownership and recurrent composite tokens.

Raw visible label zero means a real unassigned object. NaN is permitted only for a
padded/nonexistent object. Truth beyond the selected object cap is physically present
but makes the affected branch unreconstructable from the model input.
