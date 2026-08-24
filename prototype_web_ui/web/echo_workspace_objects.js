(function () {
  // Centralized model for "externalized work" -- one PRIMARY WORK SURFACE
  // (whatever the user is actually working on right now), zero or more
  // small REFERENCE objects, and Echo. See docs/ECHO_CHECKPOINT.md "Passo 3"
  // and its "one primary work surface" revision.
  //
  // Design constraints:
  // - WorkspaceRole (primary/reference/supporting/transient) is orthogonal
  //   to `type` (document/draft/comparison/...) -- the scene-management
  //   primitives below (setPrimary/demoteToReference/openTransientView/
  //   closeTransientView/recomputeLayoutIntent) never mention "document" or
  //   "draft"; only the small event-handler functions at the bottom (named
  //   after the backend's actual UI events) call into them. A future
  //   itinerary/decision-matrix/mind-map scene would reuse the same
  //   primitives without touching this file's core.
  // - Exactly one object is the "active surface" at a time (the primary,
  //   or a transient view like a comparison while it's open) -- never
  //   several large, visually-equivalent objects competing for attention.
  // - A transient view (comparison) never becomes a permanent third card;
  //   closing it removes it and returns to the primary surface.
  // - Card action buttons never call a bespoke API: they resubmit the
  //   exact natural-language phrase the backend's document-followup
  //   detector already recognizes (assistant/conversation.py), so no
  //   document logic is ever duplicated in JavaScript.
  // - Session-only state: nothing here persists across an app restart.

  const TYPE_LABELS = {
    document: "DOCUMENTO",
    draft: "RASCUNHO",
    comparison: "COMPARAÇÃO",
    excerpt: "TRECHO",
    task_artifact: "TAREFA",
    writing_task: "TAREFA",
  };

  // Phase 3B (Cognitive Workspace Blueprint v2): REPRESENTATION RENDERER.
  // Backend semantics (assistant/representation.py) already define exactly
  // these four styles -- side_by_side reuses the existing primary/co-major
  // scene primitives below (setMajorObjects) unchanged; cards/list/
  // conversation_only are flat projections with no primary/reference
  // distinction, handled by renderRepresentation() instead of the normal
  // render() path. No fifth style is ever accepted.
  const REPRESENTATION_STYLES = ["cards", "list", "side_by_side", "conversation_only"];
  const FLAT_REPRESENTATION_STYLES = ["cards", "list", "conversation_only"];

  // action id -> natural-language phrase already understood by
  // _active_document_followup_action / _looks_like_draft_refinement_request.
  // null means "handled locally, no backend round-trip".
  const ACTION_PHRASES = {
    open: null,
    close: null,
    rewrite: "torna o texto mais claro",
    improve: "ainda consegues melhorar?",
    compare: "compara as versões",
    save: "guarda esta versão",
    restore: "volta à versão anterior",
    discard: "descarta as alterações",
  };

  const ACTION_LABELS = {
    open: "Abrir",
    close: "Fechar",
    rewrite: "Melhorar",
    improve: "Melhorar",
    compare: "Comparar",
    save: "Guardar",
    restore: "Versão anterior",
    discard: "Descartar",
  };

  // Action hierarchy (section 8): the primary surface gets a handful of
  // contextual actions; anything less central is a reference and gets one.
  // Secondary actions exist but only reveal on hover/focus, never as
  // permanent buttons competing with the primary ones.
  const PRIMARY_ACTIONS = {
    document: ["rewrite"],
    draft: ["improve", "compare", "save"],
    comparison: ["close"],
    // Section 10's documented limitation: an excerpt only gets "close"
    // (return to its parent). The backend's rewrite pipeline operates on
    // the whole document/draft, not a paragraph range, so no "rewrite"
    // action is offered here that would silently apply to the wrong scope.
    excerpt: ["close"],
  };
  const PRIMARY_SECONDARY_ACTIONS = {
    draft: ["restore", "discard"],
  };
  const REFERENCE_ACTIONS = {
    document: ["open"],
    draft: ["open"],
    comparison: ["open"],
    excerpt: ["open"],
  };

  let uidCounter = 0;
  function nextId(prefix) {
    uidCounter += 1;
    return `${prefix}-${uidCounter}`;
  }

  function truncatePreview(text, max) {
    const limit = max || 220;
    const clean = String(text || "").trim();
    if (clean.length <= limit) return clean;
    return `${clean.slice(0, limit - 1).trimEnd()}…`;
  }

  function createController(options) {
    const opts = options || {};
    const container = opts.container || (typeof document !== "undefined" ? document.getElementById("workspaceObjects") : null);
    const tetherLayer = opts.tetherLayer || (typeof document !== "undefined" ? document.getElementById("workspaceTethers") : null);
    const submit = typeof opts.submit === "function" ? opts.submit : defaultSubmit;
    const getEntityPoint = typeof opts.getEntityPoint === "function" ? opts.getEntityPoint : defaultEntityPoint;
    const setEntityLayoutRole = typeof opts.setEntityLayoutRole === "function" ? opts.setEntityLayoutRole : defaultSetLayoutRole;
    // Named-anchor fallback (used only before any surface has ever
    // rendered, so there is nothing yet to measure a proximity point from).
    const setEntityAnchor = typeof opts.setEntityAnchor === "function" ? opts.setEntityAnchor : function noop() {};
    // Geometry-based placement (section 12): a real {x, y} point in
    // workspace-local coordinates, measured off the actual rendered
    // primary/transient surface via getBoundingClientRect -- not a
    // hardcoded corner. echo_ui.js supplies the real window.echoEntity.
    // setCenter(...) call; this module never touches the entity directly.
    const setEntityPoint = typeof opts.setEntityPoint === "function" ? opts.setEntityPoint : function noop() {};
    // Tells the host page whether the workspace currently owns the
    // "protagonist" role (any primary/transient surface exists), so it can
    // suppress the plain conversational response panel instead of showing
    // both at once. The backend keeps returning its usual text reply for
    // compatibility; this is presentation-only.
    const onFocusChange = typeof opts.onFocusChange === "function" ? opts.onFocusChange : function noop() {};
    // Phase 4 (section 11): the ONE frontend -> backend layout signal --
    // a semantic {pressure, visible_object_count, major_surface_count,
    // overflow_risk} object, computed fresh at the end of every render()
    // and handed to this callback ONLY when it actually changed since the
    // last report (see reportSpacePressureIfChanged). No geometry field
    // exists for a pixel/coordinate to leak into by accident.
    const reportCapacity = typeof opts.reportCapacity === "function" ? opts.reportCapacity : function noop() {};

    const objects = new Map(); // id -> WorkspaceObject
    const nodes = new Map(); // id -> HTMLElement (diffed, not rebuilt every render)
    let renderQueued = false;
    let dragState = null;
    let hadSurface = false;
    // Phase 3B: the current RepresentationState mirror, or null before the
    // first representation_changed event ever arrives -- in that default
    // state, render() behaves EXACTLY as it did before this phase existed
    // (see render()'s branch below). Session-only, never persisted, same
    // convention as `scene`.
    let representation = null;
    // Phase 4 (section 1/17): the current AttentionState mirror -- answers
    // "how important is this object right now", kept completely separate
    // from `representation` above (section 3's invariant: representation
    // answers "how should it be shown"). `objectStates` keys are semantic
    // object ids, same registry as everything else here; unknown ids read
    // as "visible" (the neutral default) via attentionLevelFor() below.
    let attentionState = { objectStates: {}, focusObjectId: "", source: "auto" };
    let lastReportedCapacity = null;
    const TASK_SURFACE_MAX_OPEN = 4;
    const TASK_SURFACE_STORAGE_KEY = "echo_os.task_surface.v1";
    const TASK_SURFACE_DEFAULT_RATIO = 0.36;
    let taskSurfaceWidthRatio = loadTaskSurfaceWidthRatio();
    let taskSurfaceResizeState = null;
    let taskSurfaceResizeHandle = null;

    // WorkspaceScene: the one piece of state that decides what the
    // workspace currently looks like. layoutIntent is one of "single" |
    // "primary_with_references" | "side_by_side" | "split" | "stack" |
    // "compare" | "isolate" (Passo 3.2, section 7 -- generalizes the
    // original single-primary model to allow >1 simultaneous "major"
    // object when the user explicitly asks for it; single-primary stays
    // the default). coObjectIds holds any EXTRA major objects beyond
    // primaryObjectId (empty in the default single-primary case, so every
    // pre-existing code path that only ever knew about primaryObjectId
    // keeps working unchanged). layoutSource/layoutOverride are Passo
    // 3.2 section 13 provenance: once a voice/pointer command sets an
    // explicit arrangement, automatic re-layout (recomputeLayoutIntent)
    // must not silently undo it.
    const scene = {
      primaryObjectId: "",
      coObjectIds: [],
      referenceObjectIds: [],
      supportingObjectIds: [],
      transientObjectId: "",
      layoutIntent: "single",
      layoutSource: "auto",
      layoutOverride: false,
      // Deictic/context state (section 5) -- session-only, never persisted,
      // never long-term memory. activeObjectId mirrors primaryObjectId;
      // lastReferencedObjectId also updates on excerpt/show targets that
      // don't necessarily become primary.
      activeObjectId: "",
      lastReferencedObjectId: "",
      previousPrimaryObjectId: "",
      recentObjectIds: [],
    };
    // Short scene-change history (section 6) -- capped, not persisted, not
    // filled by every render/animation (see pushSceneHistory).
    const SCENE_HISTORY_LIMIT = 8;
    const sceneHistory = [];
    // Client-side command log (section 18's scene_before/after and target
    // resolution/confidence fields) -- the WorkspaceScene lives entirely
    // here, so the backend telemetry dict has no visibility into it; this
    // is where that observability actually lives. Capped, not persisted.
    const COMMAND_LOG_LIMIT = 20;
    const commandLogEntries = [];
    // Phase 3B section 18: representation_style_applied/visible_count/
    // major_count/render_success diagnostics -- same capped, session-only,
    // inspectable-not-transmitted pattern as commandLogEntries above.
    const REPRESENTATION_LOG_LIMIT = 20;
    const representationLogEntries = [];

    function defaultSubmit(text) {
      const controller = typeof window !== "undefined" ? window.echoController : null;
      if (controller && typeof controller.submitMessage === "function") {
        controller.submitMessage(text);
      }
    }

    function defaultEntityPoint() {
      if (typeof document === "undefined" || !container) return null;
      const mind = document.getElementById("mind");
      if (!mind) return null;
      const rect = mind.getBoundingClientRect();
      const hostRect = container.getBoundingClientRect();
      return { x: rect.left + rect.width / 2 - hostRect.left, y: rect.top + rect.height / 2 - hostRect.top };
    }

    function defaultSetLayoutRole(role) {
      if (typeof window === "undefined") return;
      if (window.echoEntity && typeof window.echoEntity.setLayoutRole === "function") {
        window.echoEntity.setLayoutRole(role);
      }
    }

    // --- Registry -------------------------------------------------------

    function findByFileName(type, fileName) {
      for (const obj of objects.values()) {
        if (obj.type === type && obj.source && obj.source.fileName === fileName && obj.state !== "removed") {
          return obj;
        }
      }
      return null;
    }

    function getObject(id) {
      return objects.get(id) || null;
    }

    function listObjects() {
      return Array.from(objects.values()).filter((obj) => obj.state !== "removed");
    }

    function getScene() {
      return {
        primaryObjectId: scene.primaryObjectId,
        coObjectIds: scene.coObjectIds.slice(),
        referenceObjectIds: scene.referenceObjectIds.slice(),
        supportingObjectIds: scene.supportingObjectIds.slice(),
        transientObjectId: scene.transientObjectId,
        layoutIntent: scene.layoutIntent,
        layoutSource: scene.layoutSource,
        layoutOverride: scene.layoutOverride,
        activeObjectId: scene.activeObjectId,
        lastReferencedObjectId: scene.lastReferencedObjectId,
        previousPrimaryObjectId: scene.previousPrimaryObjectId,
        recentObjectIds: scene.recentObjectIds.slice(),
      };
    }

    // The object currently occupying the "big surface" slot -- the
    // transient view while one is open (e.g. a comparison), otherwise the
    // primary object. Generic: works the same regardless of object type.
    // Kept singular for callers that only ever care about "the one big
    // thing" (Echo's own tether anchor); activeSurfaceIds() below is the
    // plural form multi-object layouts (side_by_side/split/stack) need.
    function activeSurfaceId() {
      return scene.layoutIntent === "compare" && scene.transientObjectId ? scene.transientObjectId : scene.primaryObjectId;
    }

    // Section 7: every object currently rendered "major"/primary-sized --
    // just [primaryObjectId] in the default single-primary case (so every
    // pre-existing caller that assumed singularity still gets the exact
    // same one-element answer), plus coObjectIds when an explicit
    // side_by_side/split/stack arrangement is active.
    function activeSurfaceIds() {
      if (scene.layoutIntent === "compare" && scene.transientObjectId) return [scene.transientObjectId];
      const majors = [scene.primaryObjectId].concat(scene.coObjectIds).filter(Boolean);
      return Array.from(new Set(majors));
    }

    function touch(obj) {
      obj.metadata.updatedAt = Date.now();
    }

    // Section 5: records what the user just focused/referenced, purely as
    // session-local context for deictic resolution ("isto"/"isso"/"o
    // outro") -- never written to long-term memory.
    function recordDeixis(id, previousPrimaryId) {
      if (previousPrimaryId && previousPrimaryId !== id) scene.previousPrimaryObjectId = previousPrimaryId;
      scene.activeObjectId = id;
      scene.lastReferencedObjectId = id;
      scene.recentObjectIds = [id].concat(scene.recentObjectIds.filter((existingId) => existingId !== id)).slice(0, 5);
    }

    // Section 6: a scene snapshot cheap enough to keep several of around --
    // ids only, never full object content.
    function snapshotScene() {
      return {
        primaryObjectId: scene.primaryObjectId,
        coObjectIds: scene.coObjectIds.slice(),
        referenceObjectIds: scene.referenceObjectIds.slice(),
        layoutIntent: scene.layoutIntent,
        layoutSource: scene.layoutSource,
        layoutOverride: scene.layoutOverride,
      };
    }

    // Only pushed from voice-driven, scene-changing entry points (not
    // every internal setPrimary call, e.g. a plain draftReady revision
    // update) -- section 6 explicitly asks not to fill history with
    // animation-level noise. Skips a push that would be identical to the
    // most recent entry (no real change).
    function pushSceneHistory() {
      const snapshot = snapshotScene();
      const last = sceneHistory[sceneHistory.length - 1];
      if (last && last.primaryObjectId === snapshot.primaryObjectId && last.coObjectIds.join(",") === snapshot.coObjectIds.join(",")) {
        return;
      }
      sceneHistory.push(snapshot);
      if (sceneHistory.length > SCENE_HISTORY_LIMIT) sceneHistory.shift();
    }

    // "volta ao que estávamos a fazer" -- pops the most recent snapshot
    // that still refers to objects that actually still exist (an object
    // closed since then degrades gracefully: skip that entry, try the
    // next-older one) instead of restoring a scene pointing at nothing.
    function restoreSceneFromHistory() {
      while (sceneHistory.length) {
        const snapshot = sceneHistory.pop();
        if (!snapshot.primaryObjectId || !objects.has(snapshot.primaryObjectId)) continue;
        const validCoIds = snapshot.coObjectIds.filter((id) => objects.has(id));
        if (validCoIds.length >= 1) {
          setMajorObjects([snapshot.primaryObjectId].concat(validCoIds), { source: "voice_command", relation: snapshot.layoutIntent });
        } else {
          clearLayoutOverride();
          setPrimary(snapshot.primaryObjectId);
        }
        scheduleRender();
        return true;
      }
      return false;
    }

    // --- Scene primitives (type-agnostic) --------------------------------

    // Makes `id` the primary work surface. Whatever was primary before
    // becomes a reference instead of just disappearing -- "o utilizador
    // ainda pode olhar para aquilo que estava a fazer antes." If an
    // explicit multi-object layoutOverride is active (section 12/13) and
    // `id` is already part of that arrangement, the override is preserved
    // (re-focusing one of two side-by-side objects doesn't collapse them
    // back to one); otherwise a genuinely new primary clears the override
    // -- "scene tornar override impossível."
    function setPrimary(id) {
      if (!objects.has(id)) return;
      const previous = scene.primaryObjectId;
      const overrideSet = [previous].concat(scene.coObjectIds).filter(Boolean);
      const staysWithinOverride = scene.layoutOverride && overrideSet.includes(id);
      if (scene.layoutOverride && !staysWithinOverride) clearLayoutOverride();
      if (staysWithinOverride) {
        // Swap within the same arrangement: `id` becomes primary, whatever
        // it replaces (the old primary or a co-major) slides into
        // coObjectIds instead -- the set of majors stays the same two (or
        // more) objects, only which one is "primary" changes.
        scene.coObjectIds = overrideSet.filter((memberId) => memberId !== id);
      } else if (previous && previous !== id && objects.has(previous)) {
        demoteToReference(previous);
      }
      scene.referenceObjectIds = scene.referenceObjectIds.filter((refId) => refId !== id);
      scene.primaryObjectId = id;
      objects.get(id).workspaceRole = "primary";
      recordDeixis(id, previous);
      recomputeLayoutIntent();
    }

    function demoteToReference(id) {
      const obj = objects.get(id);
      if (!obj) return;
      obj.workspaceRole = "reference";
      if (!scene.referenceObjectIds.includes(id)) scene.referenceObjectIds.push(id);
    }

    // "minimiza o original" (section 3) -- demotes `id` to a reference
    // even when it is the CURRENT primary, unlike demoteToReference alone
    // (which only ever ran on the object setPrimary was about to replace).
    // Deliberately does not pick a specific replacement primary itself --
    // render()'s existing ensurePrimarySurface() safety net (section 9)
    // already promotes the most-recently-touched remaining object
    // generically, so this reuses that instead of duplicating the logic.
    function minimizeToReference(id) {
      const obj = objects.get(id);
      if (!obj) return;
      demoteToReference(id);
      if (scene.primaryObjectId === id) scene.primaryObjectId = "";
      scene.coObjectIds = scene.coObjectIds.filter((coId) => coId !== id);
      recomputeLayoutIntent();
    }

    function clearLayoutOverride() {
      // Every co-major the override is abandoning must fall back to being
      // a normal reference -- otherwise it goes silently untracked (not
      // primary, not a co-major, not in referenceObjectIds either) the
      // moment the arrangement ends. Whichever object is BECOMING the new
      // primary (if any) is demoted right back on the next line by the
      // caller's own logic, so demoting it here too is harmless.
      scene.coObjectIds.forEach((id) => demoteToReference(id));
      scene.layoutOverride = false;
      scene.layoutSource = "auto";
      scene.coObjectIds = [];
    }

    // Section 7: promotes 2+ objects to the "major" slot at once
    // (side_by_side/split/stack) -- generic, no per-type branching; any
    // WorkspaceObject can be one of the ids. Anything that was major
    // before but isn't in this new set is demoted to reference.
    function setMajorObjects(ids, options) {
      const opts = options || {};
      const validIds = Array.from(new Set(ids)).filter((id) => objects.has(id));
      if (validIds.length < 2) {
        if (validIds.length === 1) setPrimary(validIds[0]);
        return;
      }
      const previousMajors = activeSurfaceIds();
      previousMajors.forEach((id) => {
        if (!validIds.includes(id)) demoteToReference(id);
      });
      const [first, ...rest] = validIds;
      const previousPrimary = scene.primaryObjectId;
      scene.primaryObjectId = first;
      scene.coObjectIds = rest;
      validIds.forEach((id) => {
        objects.get(id).workspaceRole = "primary";
        scene.referenceObjectIds = scene.referenceObjectIds.filter((refId) => refId !== id);
      });
      scene.layoutIntent = opts.relation === "stack" ? "stack" : opts.relation === "split" ? "split" : "side_by_side";
      scene.layoutOverride = true;
      scene.layoutSource = opts.source || "voice_command";
      recordDeixis(first, previousPrimary);
      scheduleRender();
    }

    // Generic close (section 11): removes ANY object from the scene --
    // not just a transient view. Closing a temporary_reference (a
    // quick-look object opened via a voice command without dropping what
    // was already primary) restores the previous scene automatically,
    // implementing "fecha a fatura e volta ao que estávamos a fazer" even
    // when the user only says the first half.
    function closeObject(id) {
      if (id === scene.transientObjectId) {
        closeTransientView();
        return;
      }
      const obj = objects.get(id);
      const wasTemporary = Boolean(obj) && (Boolean(obj.metadata && obj.metadata.temporary) || obj.type === "excerpt");
      removeObject(id);
      if (wasTemporary) restoreSceneFromHistory();
    }

    // "organiza isto automaticamente" (section 12) -- the only other way
    // (besides an object closing or a new unrelated focus) an explicit
    // layoutOverride is allowed to end.
    function autoOrganize() {
      clearLayoutOverride();
      recomputeLayoutIntent();
      scheduleRender();
    }

    // Opens a transient view (today: only "compare") -- it borrows the
    // surface slot without changing who the primary object logically is.
    function openTransientView(id) {
      if (!objects.has(id)) return;
      scene.transientObjectId = id;
      objects.get(id).workspaceRole = "transient";
      scene.layoutIntent = "compare";
    }

    // Closing a transient view never leaves it behind as a permanent card
    // (section 5) -- it is removed outright, and the scene falls back to
    // whatever layoutIntent the remaining primary/reference objects imply.
    function closeTransientView() {
      if (!scene.transientObjectId) return;
      const id = scene.transientObjectId;
      scene.transientObjectId = "";
      removeObject(id);
      recomputeLayoutIntent();
    }

    function recomputeLayoutIntent() {
      if (scene.transientObjectId) {
        scene.layoutIntent = "compare";
        return;
      }
      // Section 13: an explicit voice/pointer arrangement wins over
      // automatic re-layout until something legitimately ends it
      // (clearLayoutOverride, called from setPrimary on a genuinely new
      // target, closeObject, or autoOrganize) -- never silently undone by
      // an unrelated lifecycle event's own setPrimary call.
      if (scene.layoutOverride) return;
      scene.layoutIntent = scene.referenceObjectIds.length > 0 ? "primary_with_references" : "single";
    }

    // --- Document / draft / comparison lifecycle ------------------------
    // Everything below this point is the ONLY place that knows about
    // "document"/"draft"/"comparison" -- it exists to translate the
    // backend's actual event names into calls on the generic primitives
    // above. A future non-document scene would add its own equally-thin
    // handlers instead of touching setPrimary/openTransientView/etc.

    function documentLoaded(fields) {
      const existing = findByFileName("document", fields.fileName);
      if (existing) {
        existing.content = fields.preview || existing.content;
        existing.metadata.chars = fields.chars;
        existing.metadata.contentType = fields.contentType;
        touch(existing);
        if (!scene.primaryObjectId) setPrimary(existing.id);
        scheduleRender();
        return existing;
      }
      const obj = {
        id: nextId("document"),
        type: "document",
        title: fields.fileName,
        subtitle: fields.contentType || "documento",
        content: truncatePreview(fields.preview, 240),
        source: { fileName: fields.fileName, contentType: fields.contentType },
        state: "created",
        workspaceRole: "reference",
        position: null,
        size: null,
        anchor: "workspace",
        relatedTo: null,
        revision: 0,
        metadata: { chars: fields.chars || 0, createdAt: Date.now(), updatedAt: Date.now() },
      };
      objects.set(obj.id, obj);
      if (!scene.primaryObjectId) setPrimary(obj.id);
      scheduleRender();
      return obj;
    }

    // Called on rewrite_attempt_started/rewrite_regeneration_started -- no
    // content yet, just a working placeholder. It becomes the primary
    // surface immediately: "working on it" IS the current task.
    function draftWorkingStarted(fileName) {
      const targetFileName = fileName || activeDocumentFileName();
      if (!targetFileName) return null;
      const doc = findByFileName("document", targetFileName);
      let draft = findByFileName("draft", targetFileName);
      if (!draft) {
        draft = {
          id: nextId("draft"),
          type: "draft",
          title: `${targetFileName} — draft`,
          subtitle: "a reescrever…",
          content: "",
          source: { fileName: targetFileName },
          state: "created",
          workspaceRole: "reference",
          position: null,
          size: null,
          anchor: "workspace",
          relatedTo: doc ? doc.id : null,
          revision: 0,
          metadata: { working: true, hasValidContent: false, createdAt: Date.now(), updatedAt: Date.now() },
        };
        objects.set(draft.id, draft);
      } else {
        draft.metadata.working = true;
      }
      setPrimary(draft.id);
      scheduleRender();
      return draft;
    }

    function activeDocumentFileName() {
      const primary = objects.get(scene.primaryObjectId);
      if (primary && (primary.type === "document" || primary.type === "draft")) return primary.source.fileName;
      const docs = Array.from(objects.values()).filter((obj) => obj.type === "document" && obj.state !== "removed");
      return docs.length ? docs[docs.length - 1].source.fileName : "";
    }

    // Called on document_draft_ready -- the draft now has real content and
    // becomes (or stays) the primary surface; its source document becomes
    // a reference if it isn't already.
    function draftReady(fields) {
      const doc = findByFileName("document", fields.fileName);
      let draft = findByFileName("draft", fields.fileName);
      if (!draft) {
        draft = {
          id: nextId("draft"),
          type: "draft",
          title: `${fields.fileName} — draft`,
          subtitle: "",
          content: "",
          source: { fileName: fields.fileName },
          state: "created",
          workspaceRole: "reference",
          position: null,
          size: null,
          anchor: "workspace",
          relatedTo: doc ? doc.id : null,
          revision: 0,
          metadata: { working: false, hasValidContent: false, createdAt: Date.now(), updatedAt: Date.now() },
        };
        objects.set(draft.id, draft);
      }
      draft.content = fields.content || "";
      draft.revision = fields.revision || draft.revision + 1;
      draft.subtitle = `revision ${draft.revision}`;
      draft.metadata.working = false;
      draft.metadata.hasValidContent = true;
      draft.metadata.basedOn = fields.basedOn || draft.metadata.basedOn;
      if (doc && !draft.relatedTo) draft.relatedTo = doc.id;
      touch(draft);
      setPrimary(draft.id); // idempotent if this revision's draft is already primary
      scheduleRender();
      return draft;
    }

    // rewrite_cancelled: drop the working placeholder if it never had real
    // content (and hand the primary surface back to the source document);
    // otherwise just clear the working flag and keep what's there.
    function draftCancelled(fileName) {
      const targetFileName = fileName || activeDocumentFileName();
      const draft = findByFileName("draft", targetFileName);
      if (!draft) return;
      if (!draft.metadata.hasValidContent) {
        const doc = findByFileName("document", targetFileName);
        removeObject(draft.id);
        if (doc) setPrimary(doc.id);
      } else {
        draft.metadata.working = false;
        scheduleRender();
      }
    }

    // rewrite_timeout: never destroy a previously valid draft/original --
    // just mark the attempt as failed.
    function draftTimedOut(fileName) {
      const targetFileName = fileName || activeDocumentFileName();
      const draft = findByFileName("draft", targetFileName);
      if (!draft) return;
      draft.metadata.working = false;
      draft.metadata.failed = true;
      scheduleRender();
    }

    function draftDiscarded(fileName) {
      const draft = findByFileName("draft", fileName);
      const doc = findByFileName("document", fileName);
      if (draft) removeObject(draft.id);
      if (doc) setPrimary(doc.id);
    }

    function draftRestored(fields) {
      return draftReady(fields);
    }

    function draftSaved(fields) {
      const draft = findByFileName("draft", fields.fileName);
      if (!draft) return;
      draft.metadata.savedAs = fields.outputName || "";
      touch(draft);
      scheduleRender();
    }

    // A comparison is a transient VIEW, not a third permanent card
    // (section 5) -- it borrows the surface slot and is removed outright
    // when closed (see closeTransientView / runAction("close")).
    function comparisonReady(fields) {
      const doc = findByFileName("document", fields.fileName);
      const draft = findByFileName("draft", fields.fileName);
      let comparison = findByFileName("comparison", fields.fileName);
      if (!comparison) {
        comparison = {
          id: nextId("comparison"),
          type: "comparison",
          title: `${fields.fileName} — comparação`,
          subtitle: `revision ${fields.revision || 0}`,
          content: "",
          source: { fileName: fields.fileName },
          state: "created",
          workspaceRole: "transient",
          position: null,
          size: null,
          anchor: "workspace",
          relatedTo: [doc ? doc.id : null, draft ? draft.id : null].filter(Boolean),
          revision: fields.revision || 0,
          metadata: { createdAt: Date.now(), updatedAt: Date.now() },
        };
        objects.set(comparison.id, comparison);
      }
      comparison.subtitle = `revision ${fields.revision || comparison.revision}`;
      comparison.revision = fields.revision || comparison.revision;
      comparison.metadata.originalContent = fields.originalContent || "";
      comparison.metadata.draftContent = fields.draftContent || "";
      comparison.relatedTo = [doc ? doc.id : null, draft ? draft.id : null].filter(Boolean);
      touch(comparison);
      openTransientView(comparison.id);
      scheduleRender();
      return comparison;
    }

    // Section 19: the generic entry point any future non-document scene
    // (an itinerary + flight_option + hotel + budget, a decision matrix +
    // options + criteria, a mind map + notes + sources) would use to add
    // its own objects -- goes through the exact same registry fields
    // document/draft/comparison objects already use, with NO type-specific
    // handling required elsewhere for a new type to work with setPrimary/
    // setMajorObjects/resolveTarget/applyWorkspaceCommand. Not wired to any
    // backend event (those three scenarios are deliberately not
    // implemented functionally this task -- section 19 asks only to prove
    // the primitives don't hardcode a type, not to build the scenarios).
    function createObject(fields) {
      const f = fields || {};
      const obj = {
        id: f.id || nextId(f.type || "object"),
        type: f.type || "object",
        title: f.title || "",
        subtitle: f.subtitle || "",
        content: f.content || "",
        source: f.source || null,
        state: "created",
        workspaceRole: f.workspaceRole || "reference",
        position: null,
        size: null,
        anchor: "workspace",
        relatedTo: f.relatedTo || null,
        parentObjectId: f.parentObjectId || null,
        revision: 0,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() },
      };
      if (f.presentation) obj.metadata.presentation = f.presentation;
      objects.set(obj.id, obj);
      if (f.primary) setPrimary(obj.id);
      scheduleRender();
      return obj;
    }

    // Passo 3.4: a progressive artifact for an ActiveTaskContext (e.g.
    // compose_email) -- reuses createObject entirely (no new object-
    // creation code) and, once the SAME task_id has an object, always
    // updates that one instead of creating a new card per turn (section
    // 15: "Não criar card novo a cada frase"). Type-agnostic beyond the
    // "task_artifact" label -- a future task_type reuses this unchanged.
    function upsertTaskArtifact(fields) {
      const f = fields || {};
      const stableId = f.id || "";
      const existing = listObjects().find(
        (obj) => (stableId && obj.id === stableId) || (obj.type === "task_artifact" && obj.metadata && obj.metadata.taskId === f.taskId)
      );
      if (existing) {
        if (f.title) existing.title = f.title;
        if (f.subtitle) existing.subtitle = f.subtitle;
        if (f.content) existing.content = f.content;
        if (f.presentation) existing.metadata.presentation = f.presentation;
        touch(existing);
        scheduleRender();
        return existing;
      }
      const created = createObject({id: stableId || null, type: "task_artifact", title: f.title, subtitle: f.subtitle, content: f.content, primary: true});
      if (f.presentation) created.metadata.presentation = f.presentation;
      created.metadata.taskId = f.taskId;
      return created;
    }

    // Phase 3B integration fix #2: the frontend/backend object-sync
    // bridge -- the backend's WorkSession.objects registry (assistant/
    // representation.py's SemanticObject) has no frontend equivalent of
    // its own; this is the ONE path that creates/updates the SAME
    // registry (`objects`/`nodes` above) any other object already uses,
    // keyed by the SAME id the backend assigned (never a second, parallel
    // identity). Reference role by default -- representation_changed
    // (applyRepresentation) decides arrangement/prominence afterwards,
    // this call only ever establishes identity + a minimal generic body.
    function upsertSemanticObject(fields) {
      const f = fields || {};
      if (!f.id) return null;
      const existing = objects.get(f.id);
      if (existing) {
        if (f.title) existing.title = f.title;
        if (f.kind) existing.type = f.kind;
        touch(existing);
        scheduleRender();
        return existing;
      }
      const created = createObject({id: f.id, type: f.kind || "object", title: f.title || "", workspaceRole: "reference"});
      if (f.taskId) created.metadata.taskId = f.taskId;
      return created;
    }

    // --- Excerpts / sub-objects (Passo 3.2, section 9) -------------------
    // Generic: a "window" over PART of a parent object's already-known
    // content -- never a semantic copy of the whole document. Works for
    // any object with a `.content` string (document/draft today; a future
    // spreadsheet-range/code-function/PDF-page excerpt would add its own
    // selector.kind without touching this function's shape). Splitting on
    // blank lines is a simple paragraph heuristic, not a real document
    // model -- sufficient for "mostra só o primeiro parágrafo" without a
    // more advanced parser this task doesn't call for.
    function focusExcerpt(parentId, index) {
      const parent = objects.get(parentId);
      if (!parent) return null;
      const paragraphs = String(parent.content || "").split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
      const zeroBasedIndex = Math.max(1, index || 1) - 1;
      const text = paragraphs[zeroBasedIndex] || paragraphs[0] || "";
      const resolvedIndex = paragraphs.length ? Math.min(zeroBasedIndex, paragraphs.length - 1) + 1 : index || 1;
      let excerpt = Array.from(objects.values()).find(
        (obj) => obj.type === "excerpt" && obj.parentObjectId === parentId && obj.state !== "removed"
      );
      if (!excerpt) {
        excerpt = {
          id: nextId("excerpt"),
          type: "excerpt",
          title: `${parent.title} — trecho`,
          subtitle: `parágrafo ${resolvedIndex}`,
          content: text,
          source: parent.source,
          state: "created",
          workspaceRole: "reference",
          position: null,
          size: null,
          anchor: "workspace",
          relatedTo: parentId,
          parentObjectId: parentId,
          selector: { kind: "paragraph", index: resolvedIndex },
          revision: 0,
          metadata: { createdAt: Date.now(), updatedAt: Date.now() },
        };
        objects.set(excerpt.id, excerpt);
      } else {
        excerpt.content = text;
        excerpt.subtitle = `parágrafo ${resolvedIndex}`;
        excerpt.selector = { kind: "paragraph", index: resolvedIndex };
        touch(excerpt);
      }
      setPrimary(excerpt.id);
      scheduleRender();
      return excerpt;
    }

    function removeObject(id) {
      const obj = objects.get(id);
      if (!obj) return;
      obj.state = "removed";
      if (scene.primaryObjectId === id) scene.primaryObjectId = "";
      if (scene.transientObjectId === id) scene.transientObjectId = "";
      scene.referenceObjectIds = scene.referenceObjectIds.filter((refId) => refId !== id);
      scene.supportingObjectIds = scene.supportingObjectIds.filter((refId) => refId !== id);
      objects.delete(id);
      // Do NOT drop this id from the `nodes` map here -- render()'s own
      // cleanup loop is what actually detaches the DOM element, and it
      // only finds ids still present in that map. Removing the id ahead
      // of that loop made it skip the id entirely, leaving the old node
      // connected to the DOM forever -- a real orphaned card (e.g. a
      // "closed" comparison silently staying on screen), the exact
      // "permanent third card" this model must never produce.
      scheduleRender();
    }

    function clear() {
      objects.clear();
      nodes.clear();
      scene.primaryObjectId = "";
      scene.coObjectIds = [];
      scene.referenceObjectIds = [];
      scene.supportingObjectIds = [];
      scene.transientObjectId = "";
      scene.layoutIntent = "single";
      scene.layoutSource = "auto";
      scene.layoutOverride = false;
      scene.activeObjectId = "";
      scene.lastReferencedObjectId = "";
      scene.previousPrimaryObjectId = "";
      scene.recentObjectIds = [];
      sceneHistory.length = 0;
      commandLogEntries.length = 0;
      // Phase 3B: a full workspace reset also ends any active
      // representation -- the next representation_changed event (if any)
      // starts fresh rather than inheriting a style from a previous topic.
      representation = null;
      representationLogEntries.length = 0;
      // Phase 4: same reset for attention/capacity -- a full workspace
      // reset (new topic) starts with a neutral attention slate and
      // forces the next render's capacity report to go through (not be
      // suppressed as "unchanged" against stale pre-reset numbers).
      attentionState = { objectStates: {}, focusObjectId: "", source: "auto" };
      lastReportedCapacity = null;
      removeTaskSurfaceResizeHandle();
      if (container) {
        container.innerHTML = "";
        delete container.dataset.representationStyle;
        delete container.dataset.taskSurface;
        delete container.dataset.openTaskCount;
        delete container.dataset.minimizedTaskCount;
      }
      updatePresentation();
      renderTethers();
    }

    // --- Presentation (entity role/anchor, response-panel suppression) --

    // Whenever the scene's "is there a surface right now" state changes,
    // tell the host page -- this drives both the entity's intensity
    // (layoutRole) and whether the plain response panel is suppressed.
    // Unlike the old click-to-focus toggle, this reflects the WHOLE
    // lifetime of a workspace session, not a momentary "open" click.
    function updatePresentation() {
      const taskState = taskSurfaceState();
      const hasSurface = Boolean(activeSurfaceId()) || taskState.openTasks.length > 0 || taskState.minimizedTasks.length > 0;
      setEntityLayoutRole(hasSurface ? "focus" : "normal");
      if (hasSurface !== hadSurface) {
        onFocusChange(hasSurface);
        hadSurface = hasSurface;
      }
      if (!hasSurface) setEntityAnchor("center");
    }

    // --- Actions ------------------------------------------------------

    function runAction(id, actionId) {
      const obj = objects.get(id);
      if (!obj) return;
      if (actionId === "open") {
        // "Abrir" = "isto é o que estou a trabalhar agora" -- generic
        // promotion, independent of type (a reference of any kind can
        // become the primary surface).
        if (id !== scene.primaryObjectId) setPrimary(id);
        scheduleRender();
        return;
      }
      if (actionId === "close") {
        // Generic (section 11): works for a transient comparison exactly
        // as before, but also for any other object now that "fecha X" is
        // a real workspace command -- closeObject dispatches to
        // closeTransientView itself when id is the transient one.
        closeObject(id);
        scheduleRender();
        return;
      }
      const phrase = ACTION_PHRASES[actionId];
      if (phrase) submit(phrase);
    }

    // --- WorkspaceCommand (Passo 3.2) ------------------------------------
    // Structured, type-agnostic model for a single spatial/layout command
    // (section 3). The backend (_try_workspace_command in
    // assistant/conversation.py) only recognizes the INTENT and a
    // best-effort raw target phrase -- it has no visibility into the live
    // scene. Everything here -- resolving "isto"/"a fatura"/"a segunda
    // versão" to an actual object id, and executing the action -- runs
    // entirely client-side, where the real WorkspaceScene lives.
    const WORKSPACE_COMMAND_DEFAULTS = {
      action: "",
      target: "",
      targetRef: "",
      relation: "",
      scope: "",
      size: "",
      visibility: "visible",
      region: "",
      selector: null,
      preserveCurrent: false,
      restorePolicy: "auto",
      confidence: 0,
    };

    function makeWorkspaceCommand(partial) {
      return Object.assign({}, WORKSPACE_COMMAND_DEFAULTS, partial || {});
    }

    // Section 4/5: target RESOLUTION against the live scene -- never pure
    // textual matching alone. Order below is deliberate: deictic pronouns
    // and scene-relative references ("isto", "o outro", "estes dois")
    // first (they depend on scene state, not on any object's own fields),
    // then version/ordinal hints, then a generic title/filename substring
    // match that works identically for ANY object type (a document, an
    // itinerary, a hotel, a decision_option -- no per-type branch here).
    function resolveTarget(phraseRaw) {
      const phrase = String(phraseRaw || "").trim().toLowerCase();
      if (!phrase) return null;
      const visible = listObjects();
      if (!visible.length) return null;
      const surfaceIds = activeSurfaceIds();

      if (phrase === "isto" || phrase === "isso" || phrase === "aquilo") {
        const id = scene.lastReferencedObjectId || scene.activeObjectId || scene.primaryObjectId;
        return id && objects.has(id) ? { id, confidence: 0.6 } : null;
      }
      if (phrase === "o outro" || phrase === "o anterior") {
        const previous = scene.previousPrimaryObjectId;
        if (previous && objects.has(previous) && !surfaceIds.includes(previous)) return { id: previous, confidence: 0.6 };
        const fallback = visible.find((obj) => !surfaceIds.includes(obj.id));
        return fallback ? { id: fallback.id, confidence: 0.4 } : null;
      }
      if (phrase === "estes dois" || phrase === "os dois") {
        const ids = scene.recentObjectIds.filter((id) => objects.has(id)).slice(0, 2);
        return ids.length === 2 ? { id: ids, confidence: 0.6 } : null;
      }
      if (phrase.indexOf("segunda versao") !== -1 || phrase === "draft" || phrase === "rascunho") {
        const draft = visible.find((obj) => obj.type === "draft");
        return draft ? { id: draft.id, confidence: 0.7 } : null;
      }
      if (phrase.indexOf("primeira versao") !== -1 || phrase === "original" || phrase === "o original") {
        const doc = visible.find((obj) => obj.type === "document");
        return doc ? { id: doc.id, confidence: 0.7 } : null;
      }
      if (phrase.indexOf("paragrafo") !== -1 || phrase === "excerto") {
        const excerpt = visible.find((obj) => obj.type === "excerpt");
        return excerpt ? { id: excerpt.id, confidence: 0.7 } : null;
      }
      if (phrase === "mail" || phrase === "email" || phrase === "mensagem") {
        const email = visible.find((obj) => {
          const haystack = `${obj.type || ""} ${obj.title || ""}`.toLowerCase();
          return haystack.indexOf("email") !== -1 || haystack.indexOf("mail") !== -1;
        });
        return email ? { id: email.id, confidence: 0.75 } : null;
      }
      if (phrase === "segundo ficheiro" || phrase === "primeiro ficheiro") {
        const byCreation = visible.slice().sort((a, b) => a.metadata.createdAt - b.metadata.createdAt);
        const candidate = phrase === "segundo ficheiro" ? byCreation[byCreation.length - 1] : byCreation[0];
        return candidate ? { id: candidate.id, confidence: 0.5 } : null;
      }
      // Generic title/source substring match -- the one rule every future
      // object type (itinerary/flight_option/hotel/budget/decision_option/
      // note) automatically gets for free, with no dedicated branch.
      const byTitle = visible.find((obj) => {
        const haystack = `${obj.title || ""} ${(obj.source && obj.source.fileName) || ""}`.toLowerCase();
        return haystack.indexOf(phrase) !== -1;
      });
      return byTitle ? { id: byTitle.id, confidence: 0.8 } : null;
    }

    // Section 18 (client-side half): backend telemetry cannot see the live
    // scene, so scene_before/after and target resolution/confidence are
    // logged here instead. Exposed via commandLog() for inspection/tests.
    function logCommand(command, resolution, sceneBefore) {
      commandLogEntries.push({
        command,
        resolution: resolution ? "resolved" : "unresolved",
        confidence: resolution ? resolution.confidence : 0,
        sceneBefore,
        sceneAfter: snapshotScene(),
      });
      if (commandLogEntries.length > COMMAND_LOG_LIMIT) commandLogEntries.shift();
    }

    function commandLog() {
      return commandLogEntries.slice();
    }

    // --- Phase 3B: Representation Renderer -------------------------------
    // Realizes the backend's RepresentationState (assistant/representation.py)
    // visually. Semantic interpretation (which style, which ids are visible/
    // major/hidden) is centralized in applyRepresentation() below -- the
    // ONE path any representation_changed event goes through, matching
    // section 2's requirement to avoid separate duplicated state machines
    // per style. side_by_side reuses the EXISTING setMajorObjects/
    // layoutMajorSurfaces primitives (built for "lado a lado" in an earlier
    // phase) unchanged; only cards/list/conversation_only need a new,
    // flat (no primary/reference distinction) render path -- see
    // renderRepresentation() below render().

    // Phase 3B integration fix #2 (section 8/9): representation_render_
    // success must mean the requested representation was actually
    // materialized against REAL frontend objects, not merely that
    // applyRepresentation ran without throwing. semantic_objects_requested/
    // _resolved/_rendered + semantic_object_missing_ids are what let fault
    // attribution tell apart "backend state is correct" from "frontend
    // registry is missing objects" from "layout/render itself failed".
    function logRepresentationDiagnostic(style, success, reason, counts) {
      const c = counts || {};
      const entry = {
        representation_style_applied: style,
        representation_render_success: success,
        reason: reason || "",
        semantic_objects_requested: c.requested || 0,
        semantic_objects_resolved: c.resolved || 0,
        semantic_objects_rendered: c.rendered || 0,
        semantic_object_missing_ids: c.missingIds || [],
      };
      representationLogEntries.push(entry);
      if (representationLogEntries.length > REPRESENTATION_LOG_LIMIT) representationLogEntries.shift();
      if (success) {
        console.log("[Echo UI JS] representation_render_success", entry);
      } else {
        console.warn("[Echo UI JS] representation_render_failed", entry);
      }
    }

    function representationLog() {
      return representationLogEntries.slice();
    }

    function getRepresentation() {
      return representation ? Object.assign({}, representation) : null;
    }

    // --- Phase 4: ATTENTION -----------------------------------------------
    // "How important is this object right now" -- deliberately a completely
    // separate mirror from `representation` above (section 3). The ONE
    // entry point for any attention_changed event; never touches style/
    // visible/major/hidden ids, `scene`, or obj.visibility.
    function applyAttention(payload) {
      const p = payload || {};
      attentionState = {
        objectStates: p.object_states && typeof p.object_states === "object" ? Object.assign({}, p.object_states) : {},
        focusObjectId: p.focus_object_id || "",
        source: p.source || "auto",
      };
      scheduleRender();
    }

    // Unknown ids (never assigned a level) read as "visible" -- the
    // neutral default; nothing is quietened or emphasized until the
    // backend explicitly says so.
    function attentionLevelFor(id) {
      return attentionState.objectStates[id] || "visible";
    }

    function getAttention() {
      return {
        objectStates: Object.assign({}, attentionState.objectStates),
        focusObjectId: attentionState.focusObjectId,
        source: attentionState.source,
      };
    }

    // Cross-cutting hidden/visible sync (section 6/7): reuses the SAME
    // `visibility` field "esconde isto" already uses (section 1/3 of the
    // earlier workspace-command phase), rather than inventing a second
    // hidden mechanism -- so a hide-then-restore-to-side_by_side sequence
    // (which goes through the EXISTING primary/reference render path, not
    // renderRepresentation()) still respects the backend's hidden set via
    // the code that already reads obj.visibility (positionReferenceRail,
    // updateNode).
    function applyRepresentationVisibility(hiddenIds) {
      const hiddenSet = new Set(hiddenIds || []);
      objects.forEach((obj, id) => {
        obj.visibility = hiddenSet.has(id) ? "hidden" : "visible";
      });
    }

    // The one central entry point for ANY representation_changed event
    // (section 2). Never mutates a semantic object's id/type/payload --
    // only .visibility (see above) and this module's own `representation`
    // mirror + scene.primaryObjectId/coObjectIds for the side_by_side case
    // (via the existing setMajorObjects primitive).
    function applyRepresentation(payload) {
      const p = payload || {};
      const requestedStyle = p.style;
      const style = REPRESENTATION_STYLES.indexOf(requestedStyle) !== -1 ? requestedStyle : "";
      if (!style) {
        logRepresentationDiagnostic(String(requestedStyle || ""), false, "unsupported_style", {});
        return false;
      }
      const visibleIds = Array.isArray(p.visible_object_ids) ? p.visible_object_ids.slice() : [];
      const majorIds = Array.isArray(p.major_object_ids) ? p.major_object_ids.slice() : [];
      const hiddenIds = Array.isArray(p.hidden_object_ids) ? p.hidden_object_ids.slice() : [];

      // Section 8: resolve EVERY requested id (visible + major) against
      // the frontend's own object registry BEFORE claiming anything --
      // this is the exact check that was missing when the backend/
      // frontend object-sync bridge didn't exist yet (a representation
      // naming 3 ids that had never been semantic_object_upserted here
      // used to report success anyway).
      const requestedIds = Array.from(new Set(visibleIds.concat(majorIds)));
      const missingIds = requestedIds.filter((id) => !objects.has(id));
      const resolvedCount = requestedIds.length - missingIds.length;

      representation = {
        style,
        styleParams: p.style_params && typeof p.style_params === "object" ? Object.assign({}, p.style_params) : {},
        visibleIds,
        majorIds,
        hiddenIds,
        source: p.source || "auto",
      };
      applyRepresentationVisibility(hiddenIds);

      if (style === "side_by_side") {
        const validMajors = majorIds.filter((id) => objects.has(id));
        if (validMajors.length < 1) {
          logRepresentationDiagnostic(style, false, "fewer_than_two_major_ids", {
            requested: requestedIds.length, resolved: resolvedCount, rendered: 0, missingIds,
          });
          return false;
        }
        // One major means "expand this object"; two or more means an actual
        // side-by-side arrangement. Both use the same object registry.
        if (validMajors.length === 1) {
          setPrimary(validMajors[0]);
        } else {
          setMajorObjects(validMajors, { source: representation.source, relation: "side_by_side" });
        }
        const peripheralRendered = visibleIds.filter((id) => objects.has(id) && validMajors.indexOf(id) === -1).length;
        logRepresentationDiagnostic(style, true, "", {
          requested: requestedIds.length, resolved: resolvedCount, rendered: validMajors.length + peripheralRendered, missingIds,
        });
        return true;
      }

      // cards/list/conversation_only: renderRepresentation() (called via
      // scheduleRender(), see render()'s branch) only ever builds a node
      // for an id that is BOTH requested-visible and actually present in
      // `objects` -- so "rendered" here is exactly resolvedCount, minus
      // conversation_only's own "everything is intentionally hidden" case
      // (0 requested, 0 rendered, still a success -- see this function's
      // own docstring / section 7).
      scheduleRender();
      const requestedVisibleCount = visibleIds.length;
      const renderedCount = visibleIds.filter((id) => objects.has(id)).length;
      const success = requestedVisibleCount === 0 || renderedCount > 0;
      logRepresentationDiagnostic(style, success, success ? "" : "no_requested_objects_resolved", {
        requested: requestedIds.length, resolved: resolvedCount, rendered: renderedCount, missingIds,
      });
      return success;
    }

    // Flat projection render path (section 4/5/7): no primary/reference
    // distinction -- every visible object is a peer, optionally tagged
    // "major" for cards (a subtle accent only; side_by_side is the style
    // that actually resizes majors, via the branch above). Reuses the SAME
    // `nodes` map and `buildNode`/fillNodeContent as the primary/reference
    // path (section 3: DOM nodes are reused, never duplicated per style).
    function visibleIdsForRepresentation() {
      const hiddenSet = new Set(representation.hiddenIds);
      const explicit = representation.visibleIds.length ? new Set(representation.visibleIds) : null;
      return listObjects().filter((obj) => {
        if (hiddenSet.has(obj.id)) return false;
        if (explicit) return explicit.has(obj.id);
        return true;
      });
    }

    // Phase 3C (section 3/9): content, not object kind, decides whether
    // something is a small "thought object" (a title, maybe a short
    // sentence) or genuinely needs a large surface (a document/draft with
    // real body text). Used both by the flat cards/list group sizing
    // (CSS, content-driven already) and by the compact side_by_side
    // layout below, so a short-content "document" gets the same compact
    // treatment a short-content "option" does -- never a per-kind rule.
    const COMPACT_CONTENT_THRESHOLD = 60;
    function isCompactObject(obj) {
      const textLength = String(obj.title || "").length + String(obj.content || "").length;
      return textLength <= COMPACT_CONTENT_THRESHOLD;
    }

    // Phase 3C (section 4/5): a single persistent wrapper that HOSTS the
    // flat-representation nodes, kept separate from the full-viewport
    // `container` itself -- so any border/background/sizing lives on a
    // shrink-to-fit element, never on the full-bleed #workspaceObjects box
    // (that was the "list becomes a full-height sidebar" bug: styling the
    // inset:0 container directly painted its border/background across the
    // WHOLE viewport regardless of how little content was inside it).
    let representationGroupEl = null;
    function ensureRepresentationGroup() {
      if (!container) return null;
      if (!representationGroupEl || !representationGroupEl.isConnected) {
        representationGroupEl = document.createElement("div");
        representationGroupEl.className = "workspace-representation-group";
        container.appendChild(representationGroupEl);
      }
      return representationGroupEl;
    }

    function updateRepresentationNode(node, obj, isMajor) {
      delete node.dataset.role;
      // Phase 3C micro-fix (section 2): a node that was previously one of
      // TWO compact side_by_side majors carries an inline max-height
      // (!important) plus dataset.split/compactMajor from
      // layoutCompactMajorSurfaces -- none of that is specific to cards/
      // list, so it must be cleared here too, exactly like the non-flat
      // path already clears it in layoutMajorSurfaces' own single-surface
      // branch. Left uncleared, a card returning from side_by_side stayed
      // artificially capped at 160px (and mismatched its siblings) even
      // though nothing about the cards style asked for that -- the actual
      // occupied-bounds measurement finalizeEchoPlacement relies on must
      // reflect this node's TRUE cards-style footprint, not a leftover
      // side_by_side one.
      delete node.dataset.split;
      delete node.dataset.compactMajor;
      node.dataset.representationRole = isMajor ? "major" : "peer";
      node.dataset.closable = "false";
      node.dataset.hidden = "false";
      // Phase 4 (section 17/18): a subtle, separate visual-language hook --
      // never overrides representationRole's own major/peer accent, just
      // layers on top of it (opacity/scale only, see styles.css).
      node.dataset.attention = attentionLevelFor(obj.id);
      node.style.removeProperty("left");
      node.style.removeProperty("top");
      node.style.removeProperty("width");
      node.style.removeProperty("right");
      node.style.removeProperty("max-height");
      const actions = representation.style === "list" ? [] : actionsFor(obj, false);
      fillNodeContent(node, obj, actions, []);
    }

    function resetTransientLayoutState(node) {
      if (!node) return;
      // Idempotence rule: geometry from a previous render is never
      // semantic state. Each render re-applies layout from objects,
      // representation, attention and the current viewport only.
      delete node.dataset.split;
      delete node.dataset.compactMajor;
      delete node.dataset.dragging;
      delete node.dataset.taskSurfaceSlot;
      delete node.dataset.taskSlotIndex;
      delete node.dataset.taskCount;
      delete node.dataset.minimizedRail;
      delete node.dataset.containedByTask;
      node.style.removeProperty("left");
      node.style.removeProperty("top");
      node.style.removeProperty("right");
      node.style.removeProperty("width");
      node.style.removeProperty("height");
      node.style.removeProperty("max-height");
      node.style.removeProperty("transform");
      node.style.removeProperty("margin");
    }

    function loadTaskSurfaceWidthRatio() {
      if (typeof window === "undefined" || !window.localStorage) return null;
      try {
        const raw = window.localStorage.getItem(TASK_SURFACE_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        const ratio = Number(parsed && parsed.widthRatio);
        return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
      } catch (err) {
        return null;
      }
    }

    function saveTaskSurfaceWidthRatio() {
      if (typeof window === "undefined" || !window.localStorage || !Number.isFinite(taskSurfaceWidthRatio)) return;
      try {
        window.localStorage.setItem(TASK_SURFACE_STORAGE_KEY, JSON.stringify({ widthRatio: taskSurfaceWidthRatio }));
      } catch (err) {
        /* localStorage can be disabled in embedded runtimes -- non-fatal */
      }
    }

    function taskSurfaceWidthLimits(hostRect) {
      const hostWidth = Math.max(1, Number(hostRect && hostRect.width) || 1);
      const minWidth = Math.min(360, Math.max(280, Math.floor(hostWidth * 0.26)));
      const maxWidth = Math.max(minWidth, Math.min(620, Math.floor(hostWidth * 0.44)));
      const defaultWidth = clampNumber(Math.floor(hostWidth * TASK_SURFACE_DEFAULT_RATIO), minWidth, maxWidth);
      return { minWidth, maxWidth, defaultWidth };
    }

    function clampNumber(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function taskSurfaceWidth(hostRect) {
      const hostWidth = Math.max(1, Number(hostRect && hostRect.width) || 1);
      const limits = taskSurfaceWidthLimits(hostRect);
      if (!Number.isFinite(taskSurfaceWidthRatio)) return limits.defaultWidth;
      return clampNumber(Math.floor(taskSurfaceWidthRatio * hostWidth), limits.minWidth, limits.maxWidth);
    }

    function setTaskSurfaceWidthFromPixels(width, hostRect, options = {}) {
      const hostWidth = Math.max(1, Number(hostRect && hostRect.width) || 1);
      const limits = taskSurfaceWidthLimits(hostRect);
      const nextWidth = clampNumber(Math.floor(width), limits.minWidth, limits.maxWidth);
      taskSurfaceWidthRatio = nextWidth / hostWidth;
      if (options.persist) saveTaskSurfaceWidthRatio();
      return nextWidth;
    }

    function applyTaskSurfaceBoundsToContainer(bounds) {
      if (!container || !container.style) return;
      container.style.setProperty("--task-surface-left", `${bounds.left}px`);
      container.style.setProperty("--task-surface-top", `${bounds.top}px`);
      container.style.setProperty("--task-surface-width", `${bounds.width}px`);
      container.style.setProperty("--task-surface-height", `${bounds.height}px`);
    }

    function ensureTaskSurfaceResizeHandle(bounds) {
      if (!container) return null;
      if (!taskSurfaceResizeHandle) {
        taskSurfaceResizeHandle = document.createElement("div");
        taskSurfaceResizeHandle.className = "task-surface-resize-handle";
        taskSurfaceResizeHandle.dataset.taskSurfaceResizeHandle = "true";
        taskSurfaceResizeHandle.setAttribute("role", "separator");
        taskSurfaceResizeHandle.setAttribute("aria-orientation", "vertical");
        taskSurfaceResizeHandle.setAttribute("title", "Redimensionar superfície de tarefas");
        taskSurfaceResizeHandle.addEventListener("pointerdown", startTaskSurfaceResize);
      }
      if (taskSurfaceResizeHandle.parentNode !== container) container.appendChild(taskSurfaceResizeHandle);
      if (bounds) {
        taskSurfaceResizeHandle.style.left = `${bounds.left + bounds.width - 5}px`;
        taskSurfaceResizeHandle.style.top = `${bounds.top}px`;
        taskSurfaceResizeHandle.style.height = `${bounds.height}px`;
      }
      return taskSurfaceResizeHandle;
    }

    function removeTaskSurfaceResizeHandle() {
      if (taskSurfaceResizeHandle && taskSurfaceResizeHandle.parentNode) {
        taskSurfaceResizeHandle.parentNode.removeChild(taskSurfaceResizeHandle);
      }
      taskSurfaceResizeHandle = null;
      if (container && container.style) {
        container.style.removeProperty("--task-surface-left");
        container.style.removeProperty("--task-surface-top");
        container.style.removeProperty("--task-surface-width");
        container.style.removeProperty("--task-surface-height");
      }
    }

    function isTaskSurfaceObject(obj) {
      return obj && (obj.type === "task_artifact" || obj.type === "writing_task");
    }

    function taskSurfaceObjects() {
      return listObjects().filter((obj) => isTaskSurfaceObject(obj));
    }

    function isAttentionVisible(level) {
      return level !== "minimized" && level !== "hidden";
    }

    function taskSurfaceState() {
      const tasks = taskSurfaceObjects();
      const openTasks = tasks.filter((obj) => isAttentionVisible(attentionLevelFor(obj.id))).slice(0, TASK_SURFACE_MAX_OPEN);
      const minimizedTasks = tasks.filter((obj) => attentionLevelFor(obj.id) === "minimized");
      const focusedTaskId = openTasks.find((obj) => attentionLevelFor(obj.id) === "focus")?.id || "";
      return { tasks, openTasks, minimizedTasks, focusedTaskId };
    }

    function taskSurfaceBounds(hostRect) {
      const gap = 24;
      const top = 74;
      const bottomRail = 70;
      const width = taskSurfaceWidth(hostRect);
      const bottom = Math.max(top + 240, hostRect.height - bottomRail);
      return {
        left: gap,
        top,
        width,
        height: Math.max(220, bottom - top),
      };
    }

    function taskSurfaceSlots(bounds, count) {
      const gap = 14;
      if (count <= 0) return [];
      if (count === 1) return [{ left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }];
      if (count === 2) {
        const h = (bounds.height - gap) / 2;
        return [0, 1].map((index) => ({ left: bounds.left, top: bounds.top + index * (h + gap), width: bounds.width, height: h }));
      }
      if (count === 3) {
        const h = (bounds.height - gap * 2) / 3;
        return [0, 1, 2].map((index) => ({ left: bounds.left, top: bounds.top + index * (h + gap), width: bounds.width, height: h }));
      }
      const w = (bounds.width - gap) / 2;
      const h = (bounds.height - gap) / 2;
      return [0, 1, 2, 3].map((index) => ({
        left: bounds.left + (index % 2) * (w + gap),
        top: bounds.top + Math.floor(index / 2) * (h + gap),
        width: w,
        height: h,
      }));
    }

    function placeTaskNode(node, slot) {
      node.style.setProperty("left", `${slot.left}px`, "important");
      node.style.setProperty("top", `${slot.top}px`, "important");
      node.style.setProperty("width", `${slot.width}px`, "important");
      node.style.setProperty("height", `${slot.height}px`, "important");
      node.style.setProperty("right", "auto", "important");
      node.style.setProperty("max-height", `${slot.height}px`, "important");
    }

    function placeMinimizedTaskNode(node, bounds, index) {
      const chipWidth = Math.min(210, Math.max(150, bounds.width / 2 - 8));
      const chipHeight = 38;
      const gap = 10;
      const x = bounds.left + (index % 2) * (chipWidth + gap);
      const y = bounds.top + bounds.height + 14 + Math.floor(index / 2) * (chipHeight + gap);
      placeTaskNode(node, { left: x, top: y, width: chipWidth, height: chipHeight });
    }

    function taskIdForObject(obj) {
      return obj && obj.metadata ? (obj.metadata.taskId || "") : "";
    }

    function objectBelongsToOpenTask(obj, openTasks) {
      const taskId = taskIdForObject(obj);
      if (!taskId) return false;
      return openTasks.some((task) => taskIdForObject(task) === taskId && obj.id !== task.id);
    }

    function placeEchoForTaskSurface(bounds, hostRect) {
      const entityRadius = 150;
      const rightZoneLeft = bounds.left + bounds.width + 40;
      const point = {
        x: Math.min(Math.max(rightZoneLeft + (hostRect.width - rightZoneLeft) * 0.52, rightZoneLeft + entityRadius), Math.max(entityRadius, hostRect.width - entityRadius)),
        y: Math.min(Math.max(hostRect.height * 0.38, entityRadius), Math.max(entityRadius, hostRect.height - entityRadius)),
      };
      setEntityPoint(point);
    }

    function renderTaskSurface() {
      if (!container) return false;
      const state = taskSurfaceState();
      const hasSurface = state.openTasks.length > 0 || state.minimizedTasks.length > 0;
      if (!hasSurface) {
        delete container.dataset.taskSurface;
        delete container.dataset.openTaskCount;
        delete container.dataset.minimizedTaskCount;
        removeTaskSurfaceResizeHandle();
        return false;
      }
      const hostRect = container.getBoundingClientRect();
      const bounds = taskSurfaceBounds(hostRect);
      const slots = taskSurfaceSlots(bounds, state.openTasks.length);
      applyTaskSurfaceBoundsToContainer(bounds);
      ensureTaskSurfaceResizeHandle(bounds);
      container.dataset.taskSurface = "active";
      container.dataset.openTaskCount = String(state.openTasks.length);
      container.dataset.minimizedTaskCount = String(state.minimizedTasks.length);
      delete container.dataset.representationStyle;

      const renderIds = new Set(state.tasks.map((obj) => obj.id));
      listObjects().forEach((obj) => {
        if (objectBelongsToOpenTask(obj, state.openTasks)) {
          const node = nodes.get(obj.id);
          if (node && node.parentNode) node.parentNode.removeChild(node);
          return;
        }
        if (!isTaskSurfaceObject(obj)) {
          const node = nodes.get(obj.id);
          if (node && node.parentNode) node.parentNode.removeChild(node);
        }
      });
      nodes.forEach((node, id) => {
        if (!objects.has(id)) {
          node.remove();
          nodes.delete(id);
        }
      });

      state.openTasks.forEach((obj, index) => {
        let node = nodes.get(obj.id);
        if (!node) {
          node = buildNode(obj);
          nodes.set(obj.id, node);
        }
        if (node.parentNode !== container) container.appendChild(node);
        node.dataset.role = "primary";
        node.dataset.taskSurfaceSlot = "true";
        node.dataset.taskSlotIndex = String(index);
        node.dataset.taskCount = String(state.openTasks.length);
        node.dataset.closable = "false";
        node.dataset.hidden = "false";
        node.dataset.attention = attentionLevelFor(obj.id);
        fillNodeContent(node, obj, actionsFor(obj, true), []);
        placeTaskNode(node, slots[index]);
        renderIds.delete(obj.id);
      });

      state.minimizedTasks.forEach((obj, index) => {
        let node = nodes.get(obj.id);
        if (!node) {
          node = buildNode(obj);
          nodes.set(obj.id, node);
        }
        if (node.parentNode !== container) container.appendChild(node);
        node.dataset.role = "reference";
        node.dataset.minimizedRail = "true";
        node.dataset.closable = "false";
        node.dataset.hidden = "false";
        node.dataset.attention = "minimized";
        fillNodeContent(node, obj, ["open"], []);
        placeMinimizedTaskNode(node, bounds, index);
        renderIds.delete(obj.id);
      });

      renderIds.forEach((id) => {
        const node = nodes.get(id);
        if (node && node.parentNode) node.parentNode.removeChild(node);
      });

      if (state.openTasks.length > 0) {
        placeEchoForTaskSurface(bounds, hostRect);
      } else {
        setEntityAnchor("center");
      }
      return true;
    }

    function startTaskSurfaceResize(event) {
      if (!container || !taskSurfaceResizeHandle) return;
      const hostRect = container.getBoundingClientRect();
      const startWidth = taskSurfaceWidth(hostRect);
      taskSurfaceResizeState = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startWidth,
        hostRect,
      };
      taskSurfaceResizeHandle.dataset.dragging = "true";
      try {
        taskSurfaceResizeHandle.setPointerCapture(event.pointerId);
      } catch (err) {
        /* pointer capture not available in this environment -- non-fatal */
      }
      event.preventDefault();
    }

    function continueTaskSurfaceResize(event) {
      if (!taskSurfaceResizeState || event.pointerId !== taskSurfaceResizeState.pointerId) return;
      const nextWidth = taskSurfaceResizeState.startWidth + (event.clientX - taskSurfaceResizeState.startX);
      setTaskSurfaceWidthFromPixels(nextWidth, taskSurfaceResizeState.hostRect, { persist: true });
      render();
    }

    function endTaskSurfaceResize(event) {
      if (!taskSurfaceResizeState || event.pointerId !== taskSurfaceResizeState.pointerId) return;
      if (taskSurfaceResizeHandle) {
        delete taskSurfaceResizeHandle.dataset.dragging;
        try {
          taskSurfaceResizeHandle.releasePointerCapture(event.pointerId);
        } catch (err) {
          /* non-fatal */
        }
      }
      taskSurfaceResizeState = null;
    }

    // Phase 3C final patch (section 2): the ONE place that turns "here is
    // a rectangle Echo should stay adjacent to, never overlap" into an
    // actual point -- tries the right side first, falls back below when
    // there's no room, and always clamps to the visible stage so Echo can
    // never end up clipped outside the viewport or hidden behind the
    // rectangle. `bounds` is host-relative ({left, top, right, bottom,
    // width, height}), so this works identically whether it came from a
    // live DOM measurement (updateEchoProximity/updateEchoProximityForGroup)
    // or was computed directly alongside a layout pass that just set it
    // (layoutCompactMajorSurfaces) -- the latter is the real Qt fix: a
    // getBoundingClientRect() taken moments after layoutMajorSurfaces sets
    // new inline positions cannot always be trusted to already reflect
    // them within the same synchronous pass, which was silently anchoring
    // Echo off a stale/overlapping measurement of the two side_by_side
    // majors and pushing it toward the top-left, sometimes clipped.
    function placeEchoAdjacentTo(bounds, hostRect) {
      const entityRadius = 160;
      const gap = 36;
      let point = {
        x: bounds.right + gap + entityRadius,
        y: bounds.top + Math.min(bounds.height, 320) / 2,
      };
      if (point.x + entityRadius > hostRect.width) {
        // No room to the right -- sit just below the rectangle instead,
        // still adjacent, never overlapping it.
        point = {
          x: bounds.left + Math.min(bounds.width, 320) / 2,
          y: bounds.bottom + gap + entityRadius,
        };
      }
      // Final safety clamp (section 2): whatever the two branches above
      // computed, never let it place Echo's own circle partially outside
      // the stage in either axis.
      point = {
        x: Math.min(Math.max(point.x, entityRadius), Math.max(entityRadius, hostRect.width - entityRadius)),
        y: Math.min(Math.max(point.y, entityRadius), Math.max(entityRadius, hostRect.height - entityRadius)),
      };
      setEntityPoint(point);
    }

    // Phase 3C (section 7): Echo's placement for a FLAT representation --
    // measured off the group wrapper's actual rect (not a specific
    // surface, there is no "primary" in a flat style), mirroring
    // updateEchoProximity()'s own adjacent-not-overlapping logic exactly.
    // An empty group (conversation_only, or cards/list with nothing
    // visible) recovers Echo to the same safe "center" anchor the
    // original no-surface case already used -- never leaves it wherever
    // it happened to be.
    function updateEchoProximityForGroup(groupEl) {
      if (!container) return;
      const rect = groupEl ? groupEl.getBoundingClientRect() : null;
      // Section 7 (conversation_only recovery): a degenerate rect in
      // EITHER dimension means there is no real content to anchor next
      // to -- an empty flex group commonly collapses to zero HEIGHT while
      // still reporting its parent's full WIDTH, so checking only "both
      // are zero" missed this and left Echo anchored to stale geometry.
      if (!rect || rect.width === 0 || rect.height === 0) {
        setEntityAnchor("center");
        return;
      }
      const hostRect = container.getBoundingClientRect();
      placeEchoAdjacentTo(
        {
          left: rect.left - hostRect.left,
          top: rect.top - hostRect.top,
          right: rect.right - hostRect.left,
          bottom: rect.bottom - hostRect.top,
          width: rect.width,
          height: rect.height,
        },
        hostRect,
      );
    }

    function renderRepresentation() {
      if (!container) return;
      const style = representation.style;
      container.dataset.representationStyle = style;
      const group = ensureRepresentationGroup();
      const visible = visibleIdsForRepresentation();
      const visibleIdSet = new Set(visible.map((obj) => obj.id));
      const majorSet = new Set(representation.majorIds);

      // Registry-level removals (an object actually closed/removed, not
      // merely hidden by this representation) still get their node
      // detached, same invariant as the normal render() cleanup loop.
      nodes.forEach((node, id) => {
        if (!objects.has(id)) {
          node.remove();
          nodes.delete(id);
        }
      });

      visible.forEach((obj) => {
        let node = nodes.get(obj.id);
        if (!node) {
          node = buildNode(obj);
          nodes.set(obj.id, node);
        }
        updateRepresentationNode(node, obj, majorSet.has(obj.id));
        if (group) group.appendChild(node); // reuses the existing node; appendChild on an attached node just reorders it
      });

      // Anything not currently visible in THIS representation (hidden by
      // conversation_only, or simply excluded) is detached from the DOM
      // but stays in `objects`/`nodes` -- never deleted (section 7).
      objects.forEach((obj, id) => {
        if (visibleIdSet.has(id)) return;
        const node = nodes.get(id);
        if (node && node.parentNode) node.parentNode.removeChild(node);
      });

      // Echo placement + tethers are handled by render()'s own shared tail
      // (finalizeEchoPlacement) -- not here. See that function's comment
      // for why a per-style tail was the actual bug.
    }

    // Section 19: dispatches purely on `action` + resolved target(s) --
    // never a per-type branch. A future itinerary/decision_option scene
    // reuses this unchanged; only resolveTarget's title-matching already
    // covers it generically.
    function applyWorkspaceCommand(rawCommand) {
      const command = makeWorkspaceCommand(rawCommand);
      const sceneBefore = snapshotScene();
      if (command.scope === "all") {
        objects.forEach((obj) => {
          obj.visibility = "visible";
          if (obj.attention === "minimized" || obj.attention === "glanceable" || obj.attention === "hidden") {
            obj.attention = "active";
          }
        });
        logCommand(command, { id: listObjects().map((obj) => obj.id), confidence: 1 }, sceneBefore);
        scheduleRender();
        return;
      }
      const resolved = command.targetRef ? { id: command.targetRef, confidence: 1 } : resolveTarget(command.target);
      const targetIds = resolved && Array.isArray(resolved.id) ? resolved.id.filter((id) => objects.has(id)) : [];
      const targetId = resolved && !Array.isArray(resolved.id) && objects.has(resolved.id) ? resolved.id : (targetIds[0] || "");

      switch (command.action) {
        case "focus":
        case "bring_forward":
        case "show": {
          if (!targetId) break;
          pushSceneHistory();
          // Section 11: "show" (a quick-look phrasing, e.g. "mostra-me o
          // ficheiro da fatura"/"preciso de ver rapidamente...") is tagged
          // temporary by default -- unlike "focus"/"bring_forward" ("vamos
          // trabalhar na segunda versão"), which means "this is now what
          // I'm doing" and stays untagged unless explicitly marked. A
          // separate metadata flag, not workspaceRole -- setPrimary below
          // always overwrites workspaceRole to "primary" while an object
          // is the active surface, so tagging that same field here would
          // be clobbered the instant this command runs.
          if (command.preserveCurrent || command.action === "show") {
            const obj = objects.get(targetId);
            if (obj) obj.metadata.temporary = true;
          }
          setPrimary(targetId);
          break;
        }
        case "focus_excerpt": {
          const parentId = targetId || scene.primaryObjectId;
          const index = (command.selector && Number(command.selector.index)) || 1;
          if (parentId) focusExcerpt(parentId, index);
          break;
        }
        case "expand": {
          if (targetId) setPrimary(targetId);
          break;
        }
        case "minimize":
        case "compact": {
          if (targetId) minimizeToReference(targetId);
          break;
        }
        case "hide": {
          const obj = targetId ? objects.get(targetId) : null;
          if (obj) obj.visibility = "hidden";
          break;
        }
        case "close": {
          if (targetId) closeObject(targetId);
          break;
        }
        case "keep_open": {
          const ids = targetIds.length ? targetIds : [targetId].filter(Boolean);
          ids.forEach((id) => {
            const obj = objects.get(id);
            if (obj) obj.metadata.keepOpen = true;
          });
          break;
        }
        case "place":
        case "place_beside":
        case "stack": {
          pushSceneHistory();
          const pairIds = targetIds.length >= 2 ? targetIds : [scene.primaryObjectId, targetId].filter(Boolean);
          if (pairIds.length >= 2) {
            setMajorObjects(pairIds, { source: "voice_command", relation: command.action === "stack" ? "stack" : "side_by_side" });
          } else if (targetId) {
            const obj = objects.get(targetId);
            if (obj) obj.visibility = "visible";
          }
          break;
        }
        case "isolate": {
          if (targetId) {
            pushSceneHistory();
            setPrimary(targetId);
          }
          break;
        }
        case "compare": {
          // Deliberately a no-op here -- the existing content-comparison
          // pipeline (document_comparison_ready -> comparisonReady) already
          // owns this; duplicating it would create two divergent paths.
          break;
        }
        case "restore_previous_scene": {
          restoreSceneFromHistory();
          break;
        }
        case "auto_organize": {
          autoOrganize();
          break;
        }
        default:
          break;
      }
      logCommand(command, resolved, sceneBefore);
      scheduleRender();
    }

    // --- Drag -----------------------------------------------------------
    // Reference objects (small, in the rail) are draggable like before.
    // The primary/transient surface has a fixed slot -- dragging it would
    // fight the very positioning that makes it legible as THE surface, so
    // it is intentionally not draggable.

    function beginDrag(id, event) {
      const obj = objects.get(id);
      const node = nodes.get(id);
      if (!obj || !node || !container || activeSurfaceIds().includes(id)) return;
      const hostRect = container.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      dragState = {
        id,
        pointerId: event.pointerId,
        offsetX: event.clientX - nodeRect.left,
        offsetY: event.clientY - nodeRect.top,
        hostRect,
      };
      try {
        node.setPointerCapture(event.pointerId);
      } catch (err) {
        /* pointer capture not available in this environment -- non-fatal */
      }
      node.dataset.dragging = "true";
      event.preventDefault();
    }

    function continueDrag(event) {
      if (!dragState || event.pointerId !== dragState.pointerId) return;
      const obj = objects.get(dragState.id);
      if (!obj) return;
      const x = event.clientX - dragState.hostRect.left - dragState.offsetX;
      const y = event.clientY - dragState.hostRect.top - dragState.offsetY;
      obj.position = { x: Math.max(0, x), y: Math.max(0, y) };
      applyPosition(dragState.id);
      renderTethers();
    }

    function endDrag(event) {
      if (!dragState || event.pointerId !== dragState.pointerId) return;
      const node = nodes.get(dragState.id);
      if (node) {
        delete node.dataset.dragging;
        try {
          node.releasePointerCapture(event.pointerId);
        } catch (err) {
          /* non-fatal */
        }
      }
      dragState = null;
    }

    // A manually dragged reference keeps its place; an auto-placed one is
    // positioned by positionReferenceRail() instead (measured off the
    // primary surface's actual rect, not a guessed offset -- section 12).
    function applyPosition(id) {
      if (activeSurfaceIds().includes(id)) return; // a major surface has its own CSS/geometry slot, no inline position
      const node = nodes.get(id);
      const obj = objects.get(id);
      if (!node || !obj || !obj.position) return;
      node.style.left = `${obj.position.x}px`;
      node.style.top = `${obj.position.y}px`;
    }

    function canonicalSurfaceBounds(hostRect) {
      const gap = 24;
      const echoReserve = hostRect.width <= 860 ? 0 : 374;
      const left = gap;
      const top = 72;
      const right = Math.max(left + 240, hostRect.width - echoReserve - gap);
      const bottom = Math.min(hostRect.height - gap, top + Math.min(660, Math.max(260, hostRect.height * 0.72)));
      return {
        left,
        top,
        right,
        bottom,
        width: Math.max(240, right - left),
        height: Math.max(180, bottom - top),
      };
    }

    // Section 14 (generic geometry engine, no advanced solver): when more
    // than one object is "major" at once (side_by_side/split/stack), gives
    // each an equal share of the same safe area the single-primary CSS
    // rule already reserves -- measured off whichever major object was
    // already rendered under that CSS rule a moment ago (real
    // getBoundingClientRect geometry, not a hardcoded column width).
    // With exactly one major, clears any leftover inline override so the
    // plain CSS [data-role="primary"] rule is back in sole control --
    // identical to how this always worked before this feature existed.
    function layoutMajorSurfaces(surfaceIds, hostRect) {
      if (surfaceIds.length < 2) {
        surfaceIds.forEach((id) => {
          const node = nodes.get(id);
          if (!node) return;
          node.style.removeProperty("left");
          node.style.removeProperty("top");
          node.style.removeProperty("width");
          node.style.removeProperty("right");
          node.style.removeProperty("max-height");
          node.dataset.split = "false";
          delete node.dataset.compactMajor;
        });
        return {echoPlaced: false, bottomEdge: null}; // caller falls back to updateEchoProximity()
      }
      // Phase 3C (section 6/9): content decides size, not "there is extra
      // viewport space to fill." Two short-content majors (e.g. generic
      // option titles) get a small, centered, bounded pair of surfaces
      // instead of stretching across the whole reserved document-editing
      // width -- real documents/drafts (their usual, longer content)
      // still get the original full-width split below, unchanged.
      const allCompact = surfaceIds.every((id) => {
        const obj = objects.get(id);
        return obj && isCompactObject(obj);
      });
      if (allCompact) {
        // Places Echo itself (from the SAME computed geometry, not a
        // re-measurement) and returns true -- see layoutCompactMajorSurfaces
        // and placeEchoAdjacentTo's own comments for why (section 2).
        return layoutCompactMajorSurfaces(surfaceIds, hostRect);
      }
      const bounds = canonicalSurfaceBounds(hostRect);
      const top = bounds.top;
      const left0 = bounds.left;
      const rightEdge = bounds.right;
      const margin = 12;
      const availableWidth = Math.max(240, rightEdge - left0);
      const stack = scene.layoutIntent === "stack";
      surfaceIds.forEach((id, index) => {
        const node = nodes.get(id);
        if (!node) return;
        node.dataset.split = "true";
        delete node.dataset.compactMajor;
        node.style.removeProperty("max-height");
        if (stack) {
          node.style.setProperty("left", `${left0}px`, "important");
          node.style.setProperty("top", `${top + index * 32}px`, "important");
          node.style.setProperty("width", `${availableWidth}px`, "important");
        } else {
          const colWidth = (availableWidth - margin * (surfaceIds.length - 1)) / surfaceIds.length;
          node.style.setProperty("left", `${left0 + index * (colWidth + margin)}px`, "important");
          node.style.setProperty("top", `${top}px`, "important");
          node.style.setProperty("width", `${colWidth}px`, "important");
        }
        node.style.setProperty("right", "auto", "important");
      });
      return {echoPlaced: false, bottomEdge: null}; // non-compact path -- caller falls back to updateEchoProximity() unchanged
    }

    // Phase 3C compact pair/group layout: bounded width/height, centered
    // within the safe cognitive area (the same area the primary surface
    // rule already reserves, i.e. leaving room for Echo on the right),
    // rather than measuring/stretching to an existing large surface rect.
    function layoutCompactMajorSurfaces(surfaceIds, hostRect) {
      const cardWidth = 240;
      const cardHeight = 160;
      const gap = 20;
      const echoReserve = 374; // same reserved-for-Echo width the primary-surface CSS rule already uses
      const usableWidth = Math.max(cardWidth, hostRect.width - echoReserve - 48);
      const totalWidth = cardWidth * surfaceIds.length + gap * (surfaceIds.length - 1);
      const startX = Math.max(24, (usableWidth - totalWidth) / 2 + 24);
      const centerY = Math.max(72, hostRect.height * 0.38);
      surfaceIds.forEach((id, index) => {
        const node = nodes.get(id);
        if (!node) return;
        node.dataset.split = "true";
        node.dataset.compactMajor = "true";
        node.style.setProperty("left", `${startX + index * (cardWidth + gap)}px`, "important");
        node.style.setProperty("top", `${centerY}px`, "important");
        node.style.setProperty("width", `${cardWidth}px`, "important");
        node.style.setProperty("max-height", `${cardHeight}px`, "important");
        node.style.setProperty("right", "auto", "important");
      });
      // Phase 3C final patch (section 2): place Echo from the SAME
      // geometry just computed above -- never by re-measuring
      // getBoundingClientRect() a moment later (the real Qt clipping bug:
      // that measurement could not be trusted to already reflect the
      // inline styles just set on the two major nodes within the same
      // synchronous pass, so Echo ended up anchored off a stale, often
      // overlapping, rect and drifted toward the top-left/viewport edge).
      const rightEdge = startX + totalWidth;
      const bottomEdge = centerY + cardHeight;
      placeEchoAdjacentTo(
        {left: startX, top: centerY, right: rightEdge, bottom: bottomEdge, width: totalWidth, height: cardHeight},
        hostRect,
      );
      // Echo placement handled here (caller must NOT also call
      // updateEchoProximity()); bottomEdge is also handed back so
      // positionReferenceRail() can place the reference rail below the
      // KNOWN compact geometry instead of re-measuring it too.
      return {echoPlaced: true, bottomEdge};
    }

    // Reference rail: a left-to-right cascade placed just BELOW the
    // lowest major surface's measured bottom edge, never guessed
    // independently of it -- a major surface can be as tall as
    // min(74vh, 660px), so a fixed "top-left" default (like the old
    // single-tier card grid used) would sit right underneath/inside it
    // instead of beside it.
    function positionReferenceRail() {
      if (!container) return null;
      const hostRect = container.getBoundingClientRect();
      const surfaceIds = activeSurfaceIds();
      const majorLayoutResult = layoutMajorSurfaces(surfaceIds, hostRect) || {};
      // Phase 3C final patch: when the compact path already computed its
      // own bottom edge, use that KNOWN value instead of re-measuring via
      // getBoundingClientRect() (the same staleness risk Echo's placement
      // had) -- falls back to the original measurement for the
      // non-compact/no-major cases, unchanged.
      let lowestBottom = majorLayoutResult.bottomEdge;
      if (lowestBottom == null) {
        lowestBottom = surfaceIds.reduce((max, id) => {
          const node = nodes.get(id);
          if (!node) return max;
          return Math.max(max, node.getBoundingClientRect().bottom - hostRect.top);
        }, 0);
      }
      const baseY = lowestBottom ? lowestBottom + 16 : 24;
      let column = 0;
      // Every visible object that ISN'T a current major surface goes in
      // the rail -- not just scene.referenceObjectIds. While a transient
      // comparison borrows the slot, an eclipsed major is still logically
      // "primary" in the scene, but must look and sit exactly like a
      // reference, or it would keep its stale position from when it last
      // rendered as a surface (see updateNode's same isSurface-only
      // visual-role rule). Objects marked hidden (section 3's
      // `visibility`) are skipped entirely -- "esconde isto" removes them
      // from view without removing them from the registry.
      listObjects().forEach((obj) => {
        if (surfaceIds.includes(obj.id) || obj.position || obj.visibility === "hidden") return;
        const node = nodes.get(obj.id);
        if (!node) return;
        node.style.left = `${24 + column * 210}px`;
        node.style.top = `${baseY}px`;
        column += 1;
      });
      // Returns the raw layout result (not just a boolean) so render()'s
      // shared finalizeEchoPlacement tail can inspect .echoPlaced itself --
      // positionReferenceRail's OWN job is only ever the rail, never Echo.
      return majorLayoutResult;
    }

    // --- Rendering --------------------------------------------------------

    function scheduleRender() {
      updatePresentation();
      if (renderQueued) return;
      renderQueued = true;
      const raf = typeof window !== "undefined" && window.requestAnimationFrame ? window.requestAnimationFrame : (fn) => setTimeout(fn, 16);
      raf(() => {
        renderQueued = false;
        render();
      });
    }

    function buildBody(obj) {
      if (obj.metadata && obj.metadata.presentation) return "";
      if (obj.type === "comparison") {
        const original = truncatePreview(obj.metadata.originalContent, 600);
        const draftText = truncatePreview(obj.metadata.draftContent, 600);
        return (
          '<div class="workspace-object__comparison">' +
          `<div class="workspace-object__comparison-col"><span>ORIGINAL</span><p>${escapeHtml(original)}</p></div>` +
          `<div class="workspace-object__comparison-col"><span>DRAFT</span><p>${escapeHtml(draftText)}</p></div>` +
          "</div>"
        );
      }
      if (obj.type === "draft" && obj.metadata.working) {
        return '<p class="workspace-object__working">A reescrever…</p>';
      }
      return `<p class="workspace-object__text">${escapeHtml(truncatePreview(obj.content, obj.workspaceRole === "primary" ? 2000 : 140))}</p>`;
    }

    function fillBody(host, obj) {
      while (host.firstChild || (host.children && host.children.length)) {
        host.removeChild(host.firstChild || host.children[0]);
      }
      const presentation = obj && obj.metadata ? obj.metadata.presentation : null;
      if (presentation && Array.isArray(presentation.sections)) {
        host.appendChild(buildCognitiveTaskProjection(obj, presentation));
        return;
      }
      const wrapper = document.createElement("div");
      wrapper.innerHTML = buildBody(obj);
      while (wrapper.firstChild) host.appendChild(wrapper.firstChild);
    }

    function buildCognitiveTaskProjection(obj, presentation) {
      const root = document.createElement("div");
      root.className = "workspace-object__cognitive";
      root.dataset.stage = presentation.stage || "";

      const visibleSections = presentation.sections.filter((section) => section && (section.text || (Array.isArray(section.items) && section.items.length)));
      visibleSections.forEach((section) => {
        const sectionNode = document.createElement("section");
        sectionNode.className = "workspace-object__cognitive-section";
        sectionNode.dataset.role = section.role || "section";
        sectionNode.dataset.priority = section.priority || "medium";

        const heading = document.createElement("h4");
        heading.textContent = section.title || "";
        sectionNode.appendChild(heading);

        if (section.text) {
          const paragraph = document.createElement("p");
          paragraph.className = "workspace-object__cognitive-text";
          paragraph.textContent = section.text;
          sectionNode.appendChild(paragraph);
        }

        if (Array.isArray(section.items) && section.items.length) {
          const list = document.createElement("ul");
          section.items.forEach((item) => {
            const itemNode = document.createElement("li");
            const label = document.createElement("span");
            label.textContent = item && item.label ? item.label : "";
            itemNode.appendChild(label);
            const details = item && Array.isArray(item.details) ? item.details.filter(Boolean) : [];
            if (details.length) {
              const detailsNode = document.createElement("small");
              detailsNode.textContent = details.join(" · ");
              itemNode.appendChild(detailsNode);
            }
            list.appendChild(itemNode);
          });
          sectionNode.appendChild(list);
        }
        root.appendChild(sectionNode);
      });

      if (!root.children.length) {
        const fallback = document.createElement("p");
        fallback.className = "workspace-object__text";
        fallback.textContent = truncatePreview(obj.content, obj.workspaceRole === "primary" ? 2000 : 140);
        root.appendChild(fallback);
      }
      return root;
    }

    function escapeHtml(text) {
      return String(text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function actionsFor(obj, isSurface) {
      if (isSurface) return (PRIMARY_ACTIONS[obj.type] || []).slice();
      return (REFERENCE_ACTIONS[obj.type] || ["open"]).slice();
    }

    function buildNode(obj) {
      const article = document.createElement("article");
      article.className = "workspace-object";
      article.dataset.id = obj.id;
      article.dataset.type = obj.type;

      const header = document.createElement("header");
      header.className = "workspace-object__header";
      header.addEventListener("pointerdown", (event) => {
        // A click on the close button (or any future header control) must
        // never be swallowed by drag-start/pointer capture.
        if (event.target.closest("[data-action]")) return;
        beginDrag(obj.id, event);
      });

      const eyebrow = document.createElement("span");
      eyebrow.className = "workspace-object__type";
      eyebrow.textContent = TYPE_LABELS[obj.type] || obj.type.toUpperCase();
      header.appendChild(eyebrow);

      // Always present, only shown via CSS for a closable surface (a
      // transient view like a comparison) -- same header-close convention
      // as #researchWorkspace/#telemetryPanel.
      const closeButton = document.createElement("button");
      closeButton.type = "button";
      closeButton.className = "workspace-object__close";
      closeButton.dataset.action = "close";
      closeButton.setAttribute("aria-label", "Fechar");
      closeButton.textContent = "×";
      header.appendChild(closeButton);
      article.appendChild(header);

      const title = document.createElement("h3");
      title.className = "workspace-object__title";
      article.appendChild(title);

      const subtitle = document.createElement("p");
      subtitle.className = "workspace-object__subtitle";
      article.appendChild(subtitle);

      const body = document.createElement("div");
      body.className = "workspace-object__body";
      article.appendChild(body);

      const actions = document.createElement("div");
      actions.className = "workspace-object__actions";
      article.appendChild(actions);

      const secondaryActions = document.createElement("div");
      secondaryActions.className = "workspace-object__actions-secondary";
      article.appendChild(secondaryActions);

      article.addEventListener("click", (event) => {
        const target = event.target.closest("[data-action]");
        if (!target) return;
        runAction(obj.id, target.dataset.action);
      });

      return article;
    }

    function fillActions(host, actionIds) {
      host.innerHTML = "";
      actionIds.forEach((actionId) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.action = actionId;
        button.textContent = ACTION_LABELS[actionId] || actionId;
        host.appendChild(button);
      });
    }

    // Shared by both render paths (the primary/reference model below and
    // Phase 3B's flat renderRepresentation()) -- title/subtitle/body/
    // actions are exactly the OBJECT RENDERER concern (section 11: how one
    // specific semantic object looks), independent of which representation
    // is currently arranging it.
    function fillNodeContent(node, obj, actionIds, secondaryActionIds) {
      node.dataset.state = obj.state;
      if (obj.metadata.working) node.dataset.working = "true";
      else delete node.dataset.working;
      node.querySelector(".workspace-object__title").textContent = obj.title;
      node.querySelector(".workspace-object__subtitle").textContent = obj.subtitle || "";
      fillBody(node.querySelector(".workspace-object__body"), obj);
      fillActions(node.querySelector(".workspace-object__actions"), actionIds);
      fillActions(node.querySelector(".workspace-object__actions-secondary"), secondaryActionIds);
    }

    function updateNode(node, obj) {
      const isSurface = activeSurfaceIds().includes(obj.id);
      // The VISUAL role is purely "is this the current surface or not" --
      // obj.workspaceRole is the logical label (the draft stays "primary"
      // in the scene model even while a comparison temporarily borrows the
      // surface slot, see openTransientView), but it must never leak into
      // rendering, or the eclipsed primary would keep looking big/primary
      // right alongside the transient view actually on screen.
      node.dataset.role = isSurface ? "primary" : "reference";
      // Mirrors the registry's lifecycle state onto the node so the CSS
      // enter-transition (.workspace-object[data-state="created"/...])
      // actually fires -- without this the node stays at the base rule's
      // opacity:0/scale(.98) forever, since only [data-role="reference"]
      // happens to carry its own opacity override (bug found in manual
      // testing: the primary/transient surface rendered with correct
      // geometry but was permanently invisible).
      node.dataset.closable = String(isSurface && (PRIMARY_ACTIONS[obj.type] || []).includes("close"));
      // "esconde isto por agora" (section 1/3's `visibility` field) --
      // stays in the registry (so "mostra X" can bring it back) but is
      // pulled from view via CSS, not removed like closeObject/removeObject.
      node.dataset.hidden = String(obj.visibility === "hidden");
      delete node.dataset.representationRole;
      node.dataset.attention = attentionLevelFor(obj.id);
      fillNodeContent(
        node,
        obj,
        actionsFor(obj, isSurface),
        isSurface ? PRIMARY_SECONDARY_ACTIONS[obj.type] || [] : []
      );
      applyPosition(obj.id);
    }

    // Section 9 safety net: never let the scene silently end up with visible
    // objects but no primary (e.g. a future bug removes the primary without
    // promoting anything else). Picks the most recently touched non-transient
    // object. This is a fallback for an invariant violation, not the normal
    // path -- normal promotion always goes through setPrimary.
    function ensurePrimarySurface() {
      if (scene.primaryObjectId && objects.has(scene.primaryObjectId)) return;
      const visible = listObjects().filter((obj) => obj.workspaceRole !== "transient");
      if (!visible.length) {
        scene.primaryObjectId = "";
        return;
      }
      const next = visible.reduce((a, b) => (b.metadata.updatedAt > a.metadata.updatedAt ? b : a));
      if (typeof console !== "undefined" && console.warn) {
        console.warn("[workspace] scene had no valid primary object; promoting", next.id);
      }
      setPrimary(next.id);
    }

    // Exposed for tests: true iff the scene invariant holds -- at most one
    // primary, and if any non-transient object exists, exactly one is
    // primary. Pure/read-only, no side effects.
    function sceneInvariantHolds() {
      const visible = listObjects().filter((obj) => obj.workspaceRole !== "transient");
      if (!visible.length) return scene.primaryObjectId === "";
      return Boolean(scene.primaryObjectId) && objects.has(scene.primaryObjectId);
    }

    function render() {
      if (!container) return;
      nodes.forEach((node) => resetTransientLayoutState(node));
      // Phase 3B: a flat representation (cards/list/conversation_only) has
      // no primary/reference concept at all -- ensurePrimarySurface()'s
      // invariant ("some object is always primary") is deliberately not
      // enforced while one of these is active. side_by_side is NOT a flat
      // style: it goes through the unchanged code below, via
      // scene.primaryObjectId/coObjectIds already set by applyRepresentation's
      // setMajorObjects call. When `representation` is null (no
      // representation_changed event has ever arrived), this branch never
      // taps and every existing document/email flow renders exactly as it
      // did before this phase.
      let majorLayoutResult = null;
      const taskSurfaceRendered = renderTaskSurface();
      if (taskSurfaceRendered) {
        majorLayoutResult = {echoPlaced: true, bottomEdge: null};
      } else if (representation && FLAT_REPRESENTATION_STYLES.indexOf(representation.style) !== -1) {
        renderRepresentation();
      } else {
        // side_by_side (or no representation at all) uses this unchanged
        // path below -- clear/refresh the container-level style marker so a
        // PRIOR flat representation's CSS (display:flex, padding, etc.)
        // never leaks into a normal primary/reference render, which would
        // otherwise shift every absolutely-positioned card by that padding.
        if (representation) container.dataset.representationStyle = representation.style;
        else delete container.dataset.representationStyle;
        ensurePrimarySurface();
        const visible = listObjects();
        const visibleIds = new Set(visible.map((obj) => obj.id));

        nodes.forEach((node, id) => {
          if (!visibleIds.has(id)) {
            node.remove();
            nodes.delete(id);
          }
        });

        visible.forEach((obj) => {
          let node = nodes.get(obj.id);
          if (!node) {
            node = buildNode(obj);
            nodes.set(obj.id, node);
          }
          // Phase 3C micro-fix (section 2/3): re-attach unconditionally,
          // not only for a brand-new node. A node that was last hosted
          // inside `.workspace-representation-group` (cards/list) or
          // detached entirely (conversation_only hid it) must come back
          // as a direct child of `container` here -- appendChild on an
          // already-correctly-attached node is a harmless no-op reorder,
          // but skipping it left a restored side_by_side surface silently
          // orphaned (never re-appended, invisible) whenever its node had
          // been built earlier under a flat style. That, in turn, is what
          // made a later return-to-cards look "wrong": the group's own
          // occupied bounds were measured correctly, but the PRECEDING
          // side_by_side state the user thought they'd restored was never
          // actually on screen to begin with.
          if (node.parentNode !== container) container.appendChild(node);
          updateNode(node, obj);
        });

        majorLayoutResult = positionReferenceRail();
      }

      // Phase 3C micro-fix (section 1): ONE shared tail, run every single
      // time render() runs, regardless of which branch above executed --
      // this is what "recalculate Echo placement on every representation
      // change" actually means structurally. Previously cards/list/
      // conversation_only recalculated via their own call inside
      // renderRepresentation() while side_by_side/single recalculated via
      // a completely separate call here; two divergent tails made it easy
      // for one of them to silently diverge (see finalizeEchoPlacement's
      // own comment) -- now there is exactly one.
      finalizeEchoPlacement(majorLayoutResult);
      renderTethers();
      reportSpacePressureIfChanged();
    }

    // Phase 4 (section 11): a small, deterministic, purely-frontend
    // capacity estimate -- object-count thresholds plus a real geometric
    // overflow check (does the representation group's measured rect
    // exceed the host), never anything the backend could mistake for a
    // pixel/coordinate. Only calls reportCapacity when the computed
    // signal actually differs from the last one sent, so a steady-state
    // workspace doesn't spam a report on every render.
    function computeSpacePressure() {
      if (!container) return null;
      const hostRect = container.getBoundingClientRect();
      const surfaceState = taskSurfaceState();
      const hasTaskSurface = surfaceState.openTasks.length > 0 || surfaceState.minimizedTasks.length > 0;
      const visibleIds = hasTaskSurface
        ? surfaceState.openTasks.map((obj) => obj.id)
        : representation
        ? visibleIdsForRepresentation().map((obj) => obj.id)
        : listObjects().filter((obj) => obj.visibility !== "hidden").map((obj) => obj.id);
      const visibleCount = visibleIds.filter((id) => attentionLevelFor(id) !== "hidden").length;
      const majorCount = hasTaskSurface ? surfaceState.openTasks.length : representation ? representation.majorIds.length : activeSurfaceIds().length;
      let pressure = "low";
      if (!hasTaskSurface && visibleCount >= 5) pressure = "high";
      else if (visibleCount >= 3) pressure = "medium";
      let overflowRisk = false;
      if (representationGroupEl && representationGroupEl.isConnected) {
        const rect = representationGroupEl.getBoundingClientRect();
        overflowRisk = rect.width > hostRect.width || rect.height > hostRect.height;
      }
      if (overflowRisk && pressure === "low") pressure = "medium";
      return {
        pressure,
        visible_object_count: visibleCount,
        major_surface_count: majorCount,
        overflow_risk: overflowRisk,
      };
    }

    function reportSpacePressureIfChanged() {
      const next = computeSpacePressure();
      if (!next) return;
      const prev = lastReportedCapacity;
      if (
        prev
        && prev.pressure === next.pressure
        && prev.visible_object_count === next.visible_object_count
        && prev.major_surface_count === next.major_surface_count
        && prev.overflow_risk === next.overflow_risk
      ) {
        return;
      }
      lastReportedCapacity = next;
      reportCapacity(next);
    }

    // Phase 3C micro-fix (section 1/6): the ONE place that decides Echo's
    // final safe position after ANY representation change -- flat
    // (cards/list/conversation_only) or not (side_by_side/single/
    // primary_with_references) -- always derived from THIS render pass's
    // final, already-settled DOM geometry (the flat branch measures the
    // real group rect; the non-flat branch reuses positionReferenceRail's
    // already-known compact-major geometry when available, falling back
    // to a live measurement otherwise). No style-specific path is allowed
    // to skip this call; it runs unconditionally at the end of render().
    function finalizeEchoPlacement(majorLayoutResult) {
      if (!container) return;
      if (representation && FLAT_REPRESENTATION_STYLES.indexOf(representation.style) !== -1) {
        updateEchoProximityForGroup(ensureRepresentationGroup());
        return;
      }
      const echoPlacedByLayout = Boolean(majorLayoutResult && majorLayoutResult.echoPlaced);
      if (!echoPlacedByLayout) updateEchoProximity();
    }

    // Section 12/14: measure the ACTUAL rendered major surface(s) instead
    // of assuming a fixed reservation. With more than one major
    // (side_by_side/split/stack), anchors off whichever one extends
    // furthest right, so Echo clears ALL of them, not just one. Falls
    // back to the named "center" anchor only when there is no surface to
    // measure yet.
    function updateEchoProximity() {
      const surfaceIds = activeSurfaceIds();
      if (!surfaceIds.length || !container) return;
      let rect = null;
      surfaceIds.forEach((id) => {
        const node = nodes.get(id);
        if (!node) return;
        const nodeRect = node.getBoundingClientRect();
        if (!rect || nodeRect.right > rect.right) rect = nodeRect;
      });
      if (!rect) return;
      const hostRect = container.getBoundingClientRect();
      placeEchoAdjacentTo(
        {
          left: rect.left - hostRect.left,
          top: rect.top - hostRect.top,
          right: rect.right - hostRect.left,
          bottom: rect.bottom - hostRect.top,
          width: rect.width,
          height: rect.height,
        },
        hostRect,
      );
    }

    // Section 5: a tether must never point at a surface that isn't actually
    // there -- not attached to the document, collapsed to zero size, or
    // effectively invisible (opacity 0, display:none). Returns null in any
    // of those cases so renderTethers skips drawing that endpoint instead of
    // drawing a "dangling" curve into empty space.
    function isNodeVisible(node) {
      if (!node || !node.isConnected) return false;
      const rect = node.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return false;
      const style = typeof getComputedStyle === "function" ? getComputedStyle(node) : null;
      if (style && (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) <= 0)) return false;
      return true;
    }

    function centerOf(id) {
      const node = nodes.get(id);
      if (!node || !isNodeVisible(node)) return null;
      const rect = node.getBoundingClientRect();
      const hostRect = container.getBoundingClientRect();
      return { x: rect.left - hostRect.left + rect.width / 2, y: rect.top - hostRect.top + rect.height / 2 };
    }

    function tetherPath(a, b) {
      if (!a || !b) return "";
      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len = Math.max(1, Math.hypot(dx, dy));
      const bow = Math.min(40, len * 0.18);
      const controlX = midX - (dy / len) * bow;
      const controlY = midY + (dx / len) * bow;
      return `M ${a.x} ${a.y} Q ${controlX} ${controlY} ${b.x} ${b.y}`;
    }

    function renderTethers() {
      if (!tetherLayer) return;
      tetherLayer.innerHTML = "";
      // A transient view (comparison) is self-contained -- no relation
      // lines to draw while it owns the surface slot.
      if (scene.layoutIntent === "compare") return;
      // Phase 3C (section 6): a representation-driven side_by_side of
      // generic thought objects carries no real "Echo is editing this"
      // relationship to show -- unlike the legacy document/entity
      // side_by_side flow (where `representation` stays null and this
      // tether keeps its original meaning), drawing it here is exactly
      // the "unnecessary tether line" the real Qt run flagged.
      const suppressEntityTether = Boolean(representation) && representation.style === "side_by_side";
      const surfaceId = suppressEntityTether ? "" : activeSurfaceId();
      const relations = [];
      objects.forEach((obj) => {
        if (obj.state === "removed" || obj.workspaceRole !== "reference") return;
        if (Array.isArray(obj.relatedTo)) {
          obj.relatedTo.forEach((relatedId) => relations.push([obj.id, relatedId]));
        } else if (obj.relatedTo) {
          relations.push([obj.id, obj.relatedTo]);
        }
      });
      if (surfaceId) {
        const entityPoint = getEntityPoint();
        const surfacePoint = centerOf(surfaceId);
        if (entityPoint && surfacePoint) {
          appendTether(entityPoint, surfacePoint, "workspace-tether--entity");
        }
      }
      relations.forEach(([fromId, toId]) => {
        const a = centerOf(fromId);
        const b = centerOf(toId);
        if (a && b) appendTether(a, b, "workspace-tether--relation");
      });
    }

    function appendTether(a, b, className) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", tetherPath(a, b));
      path.setAttribute("class", `workspace-tether ${className}`);
      tetherLayer.appendChild(path);
    }

    if (typeof document !== "undefined") {
      document.addEventListener("pointermove", continueDrag);
      document.addEventListener("pointerup", endDrag);
      document.addEventListener("pointercancel", endDrag);
      document.addEventListener("pointermove", continueTaskSurfaceResize);
      document.addEventListener("pointerup", endTaskSurfaceResize);
      document.addEventListener("pointercancel", endTaskSurfaceResize);
    }

    return {
      documentLoaded,
      draftWorkingStarted,
      draftReady,
      draftCancelled,
      draftTimedOut,
      draftDiscarded,
      draftRestored,
      draftSaved,
      comparisonReady,
      removeObject,
      clear,
      getObject,
      listObjects,
      getScene,
      activeSurfaceId,
      activeSurfaceIds,
      sceneInvariantHolds,
      runAction,
      render,
      renderTethers,
      // Passo 3.2: voice-first / natural-language workspace control.
      applyWorkspaceCommand,
      makeWorkspaceCommand,
      resolveTarget,
      commandLog,
      restoreSceneFromHistory,
      focusExcerpt,
      setMajorObjects,
      closeObject,
      autoOrganize,
      createObject,
      upsertTaskArtifact,
      // Phase 3B: Representation Renderer.
      applyRepresentation,
      getRepresentation,
      representationLog,
      // Phase 3B integration fix #2: backend/frontend object-sync bridge.
      upsertSemanticObject,
      // Phase 4: ATTENTION + SPACE PRESSURE.
      applyAttention,
      getAttention,
      computeSpacePressure,
    };
  }

  window.EchoWorkspaceObjects = { create: createController };
})();
