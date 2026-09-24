'use strict';
(() => {
  const TOOL_STORAGE = {
    quality: 'redgraft-quality-v1',
    camera: 'redgraft-camera-v1',
    motion: 'redgraft-motion-v1',
    lockSeed: 'redgraft-lock-seed-v1',
    seed: 'redgraft-seed-v1'
  };
  let lastFrameBlob = null;
  let lastFrameUrl = null;

  const baseRuntimeInput = runtimeInput;
  runtimeInput = function () {
    return { ...baseRuntimeInput(), quality_mode: $('qualityMode').value };
  };

  function randomSeed() {
    const a = new Uint32Array(2);
    crypto.getRandomValues(a);
    return (a[0] * 0x200000 + (a[1] & 0x1fffff)) % Number.MAX_SAFE_INTEGER;
  }

  function ensureSeed() {
    let value = Number($('seedValue').value);
    if (!Number.isSafeInteger(value) || value < 0) {
      value = randomSeed();
      $('seedValue').value = String(value);
    }
    safeSet(TOOL_STORAGE.seed, value);
    return value;
  }

  function cameraInstruction() {
    const camera = $('cameraMove').value;
    const motion = $('motionAmount').value;
    const cameraText = {
      none: '',
      static: 'Keep the camera locked off and stable.',
      push: 'Use a slow, smooth camera push-in.',
      pull: 'Use a slow, smooth camera pull-back.',
      pan: 'Use a smooth controlled camera pan.',
      track: 'Use a smooth tracking shot that follows the subject.',
      handheld: 'Use subtle natural handheld camera movement.',
      orbit: 'Use a smooth controlled orbit around the subject.'
    }[camera] || '';
    const motionText = {
      low: 'Keep subject motion subtle and restrained.',
      medium: 'Use natural moderate subject motion.',
      high: 'Use energetic, clearly visible subject motion while preserving temporal coherence.'
    }[motion] || '';
    return [cameraText, motionText].filter(Boolean).join(' ');
  }

  function updateLoraWarning() {
    const values = [
      'turboStrength','m3Strength','mysticStrength','hmnsfwStrength','vagassistStrength',
      'hmpussyStrength','cumshotStrength','realismStrength','deepthroatStrength',
      'civ3210503Strength','civ3320641Strength','pussy4nusStrength','fingeringStrength',
      'moawxxStrength','naughtytimesStrength'
    ].map(id => Number($(id).value) || 0);
    const active = values.filter(v => v > 0).length;
    const total = values.reduce((a,b) => a+b, 0);
    const el = $('loraWarning');
    const crowded = active >= 9 || total >= 6.5;
    el.hidden = !crowded;
    if (crowded) el.textContent = active + ' LoRAs are active (combined strength ' + total.toFixed(2) + '). If anatomy or identity gets unstable, disable specialized LoRAs you are not actively using.';
  }

  async function getInputs() {
    const url = $('imageUrl').value.trim();
    const file = url ? null : ($('image').files[0] || savedImageBlob);
    if (!file && !url) throw Error('Choose an image or enter an HTTPS image link.');
    if (file && file.size > 6000000) throw Error('Image exceeds 6 MB.');
    if (url && new URL(url).protocol !== 'https:') throw Error('Image link must use HTTPS.');
    if (lastFrameBlob && lastFrameBlob.size > 6000000) throw Error('End frame exceeds 6 MB.');
    return {
      image: file ? await fileData(file) : url,
      lastFrame: lastFrameBlob ? await fileData(lastFrameBlob) : null
    };
  }

  async function queueOne({config, image, lastFrame, prepared, settings, presetName, seed, override = null, label = ''}) {
    const runtime = runtimeInput();
    if (override) runtime[override.key] = override.value;
    const localId = uid();
    const jobSettings = { ...settings, ...runtime, seed };
    const job = {
      localId, jobId:null, endpoint:config.endpoint, status:'submitting', stage:'Submitting',
      progress:0, detail:label, prompt:prepared.used_prompt, originalPrompt:prepared.original_prompt,
      presetName: label ? presetName + ' · ' + label : presetName, settings:jobSettings, createdAt:Date.now()
    };
    jobs.push(job); persistJobs(); openDrawer('queue');
    const input = {
      image,
      prompt: prepared.used_prompt,
      length_seconds: settings.duration,
      preset_name: presetName,
      ...runtime,
      ...prepared
    };
    if (lastFrame) input.last_frame = lastFrame;
    if (seed !== null) input.seed = seed;
    const queued = await request(config, '/run', {
      input,
      policy:{executionTimeout:7200000,ttl:86400000}
    });
    if (typeof queued.id !== 'string' || !queued.id) throw Error('RunPod did not return a job ID.');
    if (!findJob(localId)) {
      try { await request(config,'/cancel/'+encodeURIComponent(queued.id),{}); } catch {}
      return;
    }
    updateJob(localId,{jobId:queued.id,status:'queued',stage:'Queued',detail:label || 'Waiting for RunPod',progress:0});
    void pollJob(localId);
  }

  submitGeneration = async function () {
    if (submitting) return;
    submitting = true;
    $('generate').disabled = true;
    try {
      const config = connection();
      const original = $('prompt').value.trim();
      if (!original) throw Error('Enter a prompt.');
      const settings = snapshotSettings();
      if (!Number.isFinite(settings.duration) || settings.duration < 0 || settings.duration > 60) throw Error('Video length must be 0–60 seconds.');
      const inputs = await getInputs();
      const direction = cameraInstruction();
      const composed = direction ? original + '\n\nCamera and motion direction: ' + direction : original;
      const prepared = await window.RedgraftPromptEnhancer.prepare(composed, false);
      if (!prepared) { message('Generation cancelled.'); return; }
      safeSet(STORAGE.prompt, original);
      safeSet(STORAGE.duration, String(settings.duration));
      const presetName = currentPresetName();
      const ab = $('abEnabled').checked;
      const seed = ($('lockSeed').checked || ab) ? ensureSeed() : null;
      if (!ab) {
        await queueOne({config,...inputs,prepared,settings,presetName,seed});
        message('Added generation to the queue.','good');
      } else {
        const key = $('abSetting').value;
        let a = Number($('abA').value), b = Number($('abB').value);
        if (!Number.isFinite(a) || !Number.isFinite(b)) throw Error('Enter valid A/B values.');
        if (key === 'steps') { a=Math.round(a); b=Math.round(b); if(a<4||a>16||b<4||b>16) throw Error('A/B steps must be 4–16.'); }
        else if (a<0||a>2||b<0||b>2) throw Error('A/B LoRA values must be 0–2.');
        await queueOne({config,...inputs,prepared,settings,presetName,seed,override:{key,value:a},label:'A · '+key+' '+a});
        await queueOne({config,...inputs,prepared,settings,presetName,seed,override:{key,value:b},label:'B · '+key+' '+b});
        message('Queued A/B pair with the same prompt, image and seed.','good');
      }
    } catch (error) {
      message(error.message,'error');
    } finally {
      submitting=false; $('generate').disabled=false;
    }
  };

  async function useBestFrame(render, button) {
    if (!render?.render_id) return;
    const old = button.textContent;
    button.disabled = true; button.textContent = 'Finding frame…';
    try {
      const out = await runAction({action:'best_frame',render_id:render.render_id});
      if (!out?.image) throw Error(out?.error || 'No reference frame was returned.');
      const blob = await (await fetch(out.image)).blob();
      savedImageBlob = blob;
      $('image').value=''; $('imageUrl').value=''; safeSet(STORAGE.imageUrl,'');
      showPreview(blob);
      await saveLocalImage(blob,'best-frame-'+render.render_id+'.jpg');
      closeDrawer();
      window.scrollTo({top:0,behavior:'smooth'});
      message('Representative frame loaded as the next first frame.','good');
    } catch (error) {
      message(error.message,'error');
    } finally {
      button.disabled=false; button.textContent=old;
    }
  }

  const baseRenderLibraryForTools = renderLibrary;
  renderLibrary = function () {
    baseRenderLibraryForTools();
    const cards=[...document.querySelectorAll('#videoLibrary .video-card')];
    cards.forEach((card,index)=>{
      if(card.querySelector('[data-best-frame]')) return;
      const render=libraryItems[index], actions=card.querySelector('.video-actions');
      if(!render?.render_id||!actions) return;
      const button=document.createElement('button');
      button.type='button';button.className='secondary';button.textContent='Use best frame';
      button.dataset.bestFrame=render.render_id;
      button.title='Pick a representative frame from this video and use it as the next first frame';
      button.addEventListener('click',()=>void useBestFrame(render,button));
      actions.prepend(button);
    });
  };

  function clearLastFrame() {
    lastFrameBlob=null;
    $('lastFrame').value='';
    if(lastFrameUrl) URL.revokeObjectURL(lastFrameUrl);
    lastFrameUrl=null; $('lastFramePreview').hidden=true;
  }

  $('qualityMode').value=safeGet(TOOL_STORAGE.quality)||'fast';
  $('cameraMove').value=safeGet(TOOL_STORAGE.camera)||'none';
  $('motionAmount').value=safeGet(TOOL_STORAGE.motion)||'medium';
  $('lockSeed').checked=(safeGet(TOOL_STORAGE.lockSeed)||'0')==='1';
  $('seedValue').value=safeGet(TOOL_STORAGE.seed)||'';
  $('qualityMode').addEventListener('change',()=>safeSet(TOOL_STORAGE.quality,$('qualityMode').value));
  $('cameraMove').addEventListener('change',()=>safeSet(TOOL_STORAGE.camera,$('cameraMove').value));
  $('motionAmount').addEventListener('change',()=>safeSet(TOOL_STORAGE.motion,$('motionAmount').value));
  $('lockSeed').addEventListener('change',()=>safeSet(TOOL_STORAGE.lockSeed,$('lockSeed').checked?'1':'0'));
  $('seedValue').addEventListener('input',()=>safeSet(TOOL_STORAGE.seed,$('seedValue').value));
  $('newSeed').addEventListener('click',()=>{$('seedValue').value=String(randomSeed());safeSet(TOOL_STORAGE.seed,$('seedValue').value)});
  $('lastFrame').addEventListener('change',()=>{const f=$('lastFrame').files[0];if(!f)return;lastFrameBlob=f;if(lastFrameUrl)URL.revokeObjectURL(lastFrameUrl);lastFrameUrl=URL.createObjectURL(f);$('lastFramePreview').src=lastFrameUrl;$('lastFramePreview').hidden=false});
  $('clearLastFrame').addEventListener('click',clearLastFrame);
  $('abEnabled').addEventListener('change',()=>{$('abControls').hidden=!$('abEnabled').checked});
  for(const id of RUNTIME_SLIDERS.map(x=>x[0])) $(id).addEventListener('input',updateLoraWarning);
  updateLoraWarning();
  renderLibrary();
})();
