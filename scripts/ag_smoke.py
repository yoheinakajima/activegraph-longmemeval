"""Gate 1 smoke test: drive the real activegraph package end-to-end."""
import activegraph as ag


def main() -> None:
    graph = ag.Graph(clock=ag.FrozenClock("2026-05-23T00:00:00Z"))
    runtime = ag.Runtime(graph, seed=0)

    graph.emit(ag.Event(id="evt-smoke-1", type="smoke.start", payload={"note": "gate-1"}))
    graph.emit(ag.Event(id="evt-smoke-2", type="smoke.tick", payload={"i": 1}))

    session = graph.add_object("Session", {"session_id": "s1", "label": "demo"})
    turn = graph.add_object("Turn", {"turn_id": "s1:t0", "role": "user", "text": "hello world"})
    rel = graph.add_relation(session.id, turn.id, "contains")

    sess_objs = graph.objects(type="Session")
    turn_objs = graph.objects(type="Turn")
    contains = graph.relations(type="contains")
    nbr_objs, nbr_rels = graph.neighborhood(session.id, depth=1)

    print(f"Sessions: {[(o.id, o.data) for o in sess_objs]}")
    print(f"Turns:    {[(o.id, o.data) for o in turn_objs]}")
    print(f"Relations(contains): {[(r.source, r.target, r.type) for r in contains]}")
    print(f"Neighborhood(session) objs={len(nbr_objs)} rels={len(nbr_rels)}")
    print(f"Runtime status events: {runtime.status(recent=10).recent_events_count if hasattr(runtime.status(recent=10), 'recent_events_count') else 'n/a'}")
    print("--- event log ---")
    for ev in graph.events:
        print(f"  {ev.id}  {ev.type}  actor={ev.actor}  ts={ev.timestamp}  payload={ev.payload}")


if __name__ == "__main__":
    main()
