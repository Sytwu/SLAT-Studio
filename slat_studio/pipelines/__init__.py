"""pipelines: high-level, user-facing entry points.

Each pipeline wires bridge/encoding + a task (edit/style/morph) + decoding into one call.
Keeps a backbone abstraction so TRELLIS vs TRELLIS.2 can be swapped behind one interface.

TODO: add EditPipeline, RestylePipeline, MorphPipeline as the tasks land.
"""
