from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"Could not find patch target: {label}")
    return text.replace(old, new, 1)


def patch_worker() -> None:
    path = Path("worker.py")
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "import model_setup\nimport runtime_health\n",
        "import model_setup\nimport runtime_health\nfrom comfy_progress import ComfyProgressTracker\nfrom runtime_controls import MINIMAX_RUNTIME_OPTION_NAMES, apply_minimax_runtime_options\n",
        "worker imports",
    )
    text = text.replace('WORKER_VERSION = "runpod-minimax-7"', 'WORKER_VERSION = "runpod-minimax-8"', 1)

    old_progress = '''def report_progress(job: dict, stage: str) -> None:\n    LOGGER.info("Job stage: %s", stage)\n    if not os.getenv("RUNPOD_POD_ID") or not job.get("id"):\n        return\n    try:\n        import runpod\n        runpod.serverless.progress_update(job, {"stage": stage})\n    except Exception:\n        LOGGER.warning("Could not send progress update")\n'''
    new_progress = '''def report_progress(job: dict, stage: str, progress: int | None = None,\n                    detail: str | None = None) -> None:\n    payload: dict[str, Any] = {"stage": stage}\n    if progress is not None:\n        payload["progress"] = max(0, min(100, int(progress)))\n    if detail:\n        payload["detail"] = detail\n    LOGGER.info("Job progress: %s", json.dumps(payload))\n    if not os.getenv("RUNPOD_POD_ID") or not job.get("id"):\n        return\n    try:\n        import runpod\n        runpod.serverless.progress_update(job, payload)\n    except Exception:\n        LOGGER.warning("Could not send progress update")\n'''
    text = replace_once(text, old_progress, new_progress, "progress reporter")

    old_allowed = '''    if set(job_input) - {"image", "prompt", "length_seconds"}:\n        return {"error": "Only input.image and input.prompt are supported, plus optional input.length_seconds"}\n'''
    new_allowed = '''    allowed_inputs = {"image", "prompt", "length_seconds", *MINIMAX_RUNTIME_OPTION_NAMES}\n    if set(job_input) - allowed_inputs:\n        return {"error": "Unsupported generation input. Use image, prompt, length_seconds, or the MiniMax runtime controls."}\n    runtime_keys = set(job_input) & MINIMAX_RUNTIME_OPTION_NAMES\n    if runtime_keys and model_setup.MODEL_PROFILE != "minimax":\n        return {"error": "MiniMax runtime controls require MODEL_PROFILE=minimax"}\n'''
    text = replace_once(text, old_allowed, new_allowed, "allowed generation inputs")

    old_config = '''        configuration = workflow_details(workflow)\n        report_progress(job, "Checking image and model files")\n'''
    new_config = '''        configuration = workflow_details(workflow)\n        if model_setup.MODEL_PROFILE == "minimax":\n            try:\n                configuration["runtime_options"] = apply_minimax_runtime_options(workflow, job_input)\n            except ValueError as exc:\n                raise WorkerError(str(exc)) from exc\n        report_progress(job, "Checking image and model files", 2)\n'''
    text = replace_once(text, old_config, new_config, "runtime option application")

    text = text.replace(
        '        wait_for_comfyui(deadline, monitor)\n        check_deadline(deadline)\n',
        '        wait_for_comfyui(deadline, monitor)\n        report_progress(job, "Worker ready; preparing generation", 6)\n        check_deadline(deadline)\n',
        1,
    )

    old_queue = '''        LOGGER.info("Requested generation configuration: %s", json.dumps(configuration))\n        report_progress(job, "Generating video from your prepared prompt")\n        prompt_id = queue_workflow(workflow, client_id=uuid.uuid4().hex)\n        history = wait_for_history(prompt_id, deadline, monitor)\n        generation_finished = True\n'''
    new_queue = '''        LOGGER.info("Requested generation configuration: %s", json.dumps(configuration))\n        report_progress(job, "Starting MiniMax generation", 8)\n        client_id = uuid.uuid4().hex\n        prompt_id = queue_workflow(workflow, client_id=client_id)\n        tracker = ComfyProgressTracker(COMFY_URL, job, prompt_id, client_id, report_progress).start()\n        try:\n            history = wait_for_history(prompt_id, deadline, monitor)\n        finally:\n            tracker.stop()\n        generation_finished = True\n'''
    text = replace_once(text, old_queue, new_queue, "ComfyUI progress tracker")
    text = text.replace('        report_progress(job, "Preparing video for download")\n', '        report_progress(job, "Preparing video for download", 99)\n', 1)

    path.write_text(text, encoding="utf-8")


def patch_docs() -> None:
    path = Path("docs/index.html")
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '.saved{font-size:12px;color:#94a0b6;margin-top:5px}\n',
        '.saved{font-size:12px;color:#94a0b6;margin-top:5px}\n.setting{margin-top:16px}.slider-head{display:flex;justify-content:space-between;align-items:center;gap:12px;font-weight:700;margin-bottom:7px}.slider-head output{font-variant-numeric:tabular-nums;color:#d8d2ff}.setting input[type="range"]{width:100%;padding:0;border:0;background:transparent;box-shadow:none;accent-color:#c4baff}.progress-panel{margin:14px 0 10px;padding:12px 13px;border:1px solid #39435a;border-radius:12px;background:#101521}.progress-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px}.progress-head strong{font-variant-numeric:tabular-nums}progress{width:100%;height:14px;accent-color:#c4baff}\n',
        "settings/progress CSS",
    )

    duration_card = '''  <div class="card">\n    <label for="duration">Video length (seconds)</label>\n    <input id="duration" type="number" min="0" max="60" step="1" inputmode="numeric" value="10">\n    <div class="hint">Choose 1–60 seconds. Set 0 to use the workflow default length.</div>\n  </div>\n'''
    settings_card = duration_card + '''\n  <div class="card">\n    <label>MiniMax generation settings</label>\n    <div class="hint">These sliders are saved automatically on this device. Set a LoRA to 0.00 to remove its effect for that generation.</div>\n\n    <div class="setting"><div class="slider-head"><span>Turbo</span><output id="turboStrengthValue">0.85</output></div><input id="turboStrength" type="range" min="0" max="1.5" step="0.05" value="0.85"></div>\n    <div class="setting"><div class="slider-head"><span>M3</span><output id="m3StrengthValue">0.50</output></div><input id="m3Strength" type="range" min="0" max="1.5" step="0.05" value="0.50"></div>\n    <div class="setting"><div class="slider-head"><span>Mystic</span><output id="mysticStrengthValue">1.00</output></div><input id="mysticStrength" type="range" min="0" max="1.5" step="0.05" value="1.00"></div>\n    <div class="setting"><div class="slider-head"><span>HMNSFW</span><output id="hmnsfwStrengthValue">1.00</output></div><input id="hmnsfwStrength" type="range" min="0" max="1.5" step="0.05" value="1.00"></div>\n    <div class="setting"><div class="slider-head"><span>vagassist</span><output id="vagassistStrengthValue">1.00</output></div><input id="vagassistStrength" type="range" min="0" max="1.5" step="0.05" value="1.00"></div>\n    <div class="setting"><div class="slider-head"><span>HMPussy</span><output id="hmpussyStrengthValue">0.35</output></div><input id="hmpussyStrength" type="range" min="0" max="1.5" step="0.05" value="0.35"></div>\n    <div class="setting"><div class="slider-head"><span>HMCumshot</span><output id="cumshotStrengthValue">0.70</output></div><input id="cumshotStrength" type="range" min="0" max="1.5" step="0.05" value="0.70"></div>\n    <div class="setting"><div class="slider-head"><span>Sampling steps</span><output id="stepsValue">8</output></div><input id="steps" type="range" min="4" max="16" step="1" value="8"></div>\n  </div>\n'''
    text = replace_once(text, duration_card, settings_card, "settings card")

    actions = '''    <div class="actions">\n      <button id="generate" type="submit">Generate Video</button>\n      <button id="resume" class="secondary" type="button" hidden>Check Saved Job</button>\n      <button id="cancel" class="danger" type="button" hidden>Cancel Job</button>\n    </div>\n'''
    actions_progress = actions + '''    <div id="progressPanel" class="progress-panel" hidden>\n      <div class="progress-head"><span id="progressStage">Starting…</span><strong id="progressValue">0%</strong></div>\n      <progress id="progressBar" max="100" value="0"></progress>\n      <div id="progressDetail" class="tiny"></div>\n    </div>\n'''
    text = replace_once(text, actions, actions_progress, "progress panel")

    old_storage = '''  imageUrl:'redgraft-image-url-v2',\n  duration:'redgraft-duration-v1',\n  job:'redgraft-job-v2'\n};\n'''
    new_storage = '''  imageUrl:'redgraft-image-url-v2',\n  duration:'redgraft-duration-v1',\n  job:'redgraft-job-v2',\n  turboStrength:'redgraft-turbo-strength-v1',\n  m3Strength:'redgraft-m3-strength-v1',\n  mysticStrength:'redgraft-mystic-strength-v1',\n  hmnsfwStrength:'redgraft-hmnsfw-strength-v1',\n  vagassistStrength:'redgraft-vagassist-strength-v1',\n  hmpussyStrength:'redgraft-hmpussy-strength-v1',\n  cumshotStrength:'redgraft-cumshot-strength-v1',\n  steps:'redgraft-steps-v1'\n};\n'''
    text = replace_once(text, old_storage, new_storage, "local storage keys")

    marker = '''let active=null,polling=false,submittingJob=false,previewUrl=null,pollEpoch=0;\nconst resultBlobs=[];\n'''
    helpers = marker + '''const RUNTIME_SLIDERS=[\n  ['turboStrength','turboStrength','0.85',2],['m3Strength','m3Strength','0.50',2],\n  ['mysticStrength','mysticStrength','1.00',2],['hmnsfwStrength','hmnsfwStrength','1.00',2],\n  ['vagassistStrength','vagassistStrength','1.00',2],['hmpussyStrength','hmpussyStrength','0.35',2],\n  ['cumshotStrength','cumshotStrength','0.70',2],['steps','steps','8',0]\n];\n'''
    text = replace_once(text, marker, helpers, "runtime slider constants")

    safe_helpers = '''function safeGet(store,key){try{return store.getItem(key)||''}catch{return ''}}\nfunction safeSet(store,key,val){try{val?store.setItem(key,val):store.removeItem(key)}catch{}}\n'''
    new_helpers = safe_helpers + '''function renderSlider(id,decimals){const input=$(id),out=$(id+'Value');if(out)out.textContent=Number(input.value).toFixed(decimals)}\nfunction bindRuntimeSliders(){for(const [id,key,fallback,decimals] of RUNTIME_SLIDERS){const input=$(id);input.value=safeGet(localStorage,STORAGE[key])||fallback;renderSlider(id,decimals);input.addEventListener('input',()=>{renderSlider(id,decimals);safeSet(localStorage,STORAGE[key],input.value)})}}\nfunction resetRuntimeSliders(){for(const [id,key,fallback,decimals] of RUNTIME_SLIDERS){safeSet(localStorage,STORAGE[key],'');$(id).value=fallback;renderSlider(id,decimals)}}\nfunction runtimeInput(){return {turbo_strength:Number($('turboStrength').value),m3_strength:Number($('m3Strength').value),mystic_strength:Number($('mysticStrength').value),hmnsfw_strength:Number($('hmnsfwStrength').value),vagassist_strength:Number($('vagassistStrength').value),hmpussy_strength:Number($('hmpussyStrength').value),cumshot_strength:Number($('cumshotStrength').value),steps:Number($('steps').value)}}\nfunction updateProgress(value,stage='',detail=''){const number=Math.max(0,Math.min(100,Math.round(Number(value)||0)));$('progressPanel').hidden=false;$('progressBar').value=number;$('progressValue').textContent=number+'%';if(stage)$('progressStage').textContent=stage;$('progressDetail').textContent=detail||''}\nfunction clearProgress(){$('progressPanel').hidden=true;$('progressBar').value=0;$('progressValue').textContent='0%';$('progressStage').textContent='Starting…';$('progressDetail').textContent=''}\n'''
    text = replace_once(text, safe_helpers, new_helpers, "runtime/progress JS helpers")

    old_poll = '''      const stage=state.output?.stage;\n      const text=stage||(state.status==='IN_QUEUE'?'Queued — waiting for an available worker.':'Running — generation is in progress.');\n      message(text);\n'''
    new_poll = '''      const stage=state.output?.stage;\n      const progress=Number(state.output?.progress);\n      const detail=state.output?.detail||'';\n      const text=stage||(state.status==='IN_QUEUE'?'Queued — waiting for an available worker.':'Running — generation is in progress.');\n      if(state.status==='IN_QUEUE')updateProgress(0,'Queued','Waiting for an available worker.');\n      else if(Number.isFinite(progress))updateProgress(progress,text,detail);\n      message(detail?text+'\\n'+detail:text);\n'''
    text = replace_once(text, old_poll, new_poll, "poll progress rendering")

    text = text.replace(
        "        try{showResult(state.output);message('Video ready.','good')}catch(error){message(error.message,'error')}finally{finish()}return;",
        "        try{updateProgress(100,'Complete');showResult(state.output);message('Video ready.','good')}catch(error){message(error.message,'error')}finally{finish()}return;",
        1,
    )

    text = text.replace(
        "  Object.values(STORAGE).forEach(k=>safeSet(localStorage,k,''));\n  active=null;$('endpoint').value='';$('key').value='';$('prompt').value='';$('imageUrl').value='';$('duration').value='10';$('image').value='';$('preview').hidden=true;$('job').textContent='';resetResults();updateBadge();message('Saved settings cleared.');controls(false);$('connection').open=true;\n",
        "  Object.values(STORAGE).forEach(k=>safeSet(localStorage,k,''));\n  resetRuntimeSliders();clearProgress();\n  active=null;$('endpoint').value='';$('key').value='';$('prompt').value='';$('imageUrl').value='';$('duration').value='10';$('image').value='';$('preview').hidden=true;$('job').textContent='';resetResults();updateBadge();message('Saved settings cleared.');controls(false);$('connection').open=true;\n",
        1,
    )

    text = text.replace(
        "    controls(true);resetResults();message('Submitting your request…');\n",
        "    controls(true);resetResults();updateProgress(0,'Submitting request');message('Submitting your request…');\n",
        1,
    )
    text = replace_once(
        text,
        "    const queued=await request(config,'/run',{input:{image,prompt,length_seconds:duration},policy:{executionTimeout:7200000,ttl:86400000}});\n",
        "    const queued=await request(config,'/run',{input:{image,prompt,length_seconds:duration,...runtimeInput()},policy:{executionTimeout:7200000,ttl:86400000}});\n",
        "runtime controls submission",
    )
    text = replace_once(
        text,
        "  $('duration').value=safeGet(localStorage,STORAGE.duration)||'10';\n  updateBadge();\n",
        "  $('duration').value=safeGet(localStorage,STORAGE.duration)||'10';\n  bindRuntimeSliders();clearProgress();\n  updateBadge();\n",
        "runtime control initialization",
    )

    path.write_text(text, encoding="utf-8")


patch_worker()
patch_docs()
print("Runtime sliders and progress tracking patch applied")
