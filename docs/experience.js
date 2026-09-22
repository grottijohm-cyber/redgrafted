'use strict';
(() => {
  const ENHANCE_STORAGE = {
    enabled: 'redgraft-prompt-enhance-enabled-v1',
    mode: 'redgraft-prompt-enhance-mode-v1',
    show: 'redgraft-prompt-enhance-show-v1'
  };
  let promptChoiceResolver = null;
  let cachedDecision = null;

  function enhancementConfig(overrides = {}) {
    const enabled = overrides.enabled ?? $('autoEnhance').checked;
    const mode = String(overrides.mode ?? $('enhanceMode').value ?? 'detailed').toLowerCase();
    const show = overrides.show ?? $('showEnhancedPrompt').checked;
    return { enabled: Boolean(enabled), mode, show: Boolean(show) };
  }

  function persistEnhancementConfig() {
    safeSet(ENHANCE_STORAGE.enabled, $('autoEnhance').checked ? '1' : '0');
    safeSet(ENHANCE_STORAGE.mode, $('enhanceMode').value);
    safeSet(ENHANCE_STORAGE.show, $('showEnhancedPrompt').checked ? '1' : '0');
  }

  async function enhanceOnly(prompt, mode = 'detailed', isExtend = false) {
    const out = await runAction({
      action: 'enhance_prompt',
      prompt,
      mode,
      is_extend: Boolean(isExtend)
    });
    if (!out?.enhanced_prompt) throw Error('The prompt enhancer returned no enhanced prompt.');
    return String(out.enhanced_prompt).trim();
  }

  function closePromptChoice(value) {
    $('promptPreviewModal').hidden = true;
    const resolve = promptChoiceResolver;
    promptChoiceResolver = null;
    if (resolve) resolve(value);
  }

  function choosePrompt(original, enhanced, contextLabel = 'generation') {
    $('promptPreviewContext').textContent = contextLabel === 'extension'
      ? 'Review the AI-enhanced continuation before it is queued.'
      : 'Review the AI-enhanced prompt before the video is queued.';
    $('previewOriginalPrompt').textContent = original;
    $('previewEnhancedPrompt').textContent = enhanced;
    $('promptPreviewModal').hidden = false;
    return new Promise(resolve => { promptChoiceResolver = resolve; });
  }

  async function preparePrompt(original, isExtend = false, overrides = {}) {
    const text = String(original || '').trim();
    if (!text) throw Error('Enter a prompt.');
    const cfg = enhancementConfig(overrides);
    if (!cfg.enabled) {
      return {
        original_prompt: text,
        enhanced_prompt: null,
        used_prompt: text,
        prompt_enhancement_enabled: false,
        prompt_enhancement_mode: cfg.mode,
        prompt_enhancement_previewed: false
      };
    }

    const cacheKey = [text, cfg.mode, cfg.show ? 'show' : 'auto', isExtend ? 'extend' : 'create'].join('\u0000');
    if (cachedDecision?.key === cacheKey) return { ...cachedDecision.value };

    message(isExtend ? 'Enhancing continuation prompt…' : 'Enhancing prompt…');
    const enhanced = await enhanceOnly(text, cfg.mode, isExtend);
    let used = enhanced;
    let previewed = false;
    if (cfg.show) {
      previewed = true;
      const choice = await choosePrompt(text, enhanced, isExtend ? 'extension' : 'generation');
      if (choice === null) return null;
      used = choice === 'original' ? text : enhanced;
    }
    const value = {
      original_prompt: text,
      enhanced_prompt: enhanced,
      used_prompt: used,
      prompt_enhancement_enabled: true,
      prompt_enhancement_mode: cfg.mode,
      prompt_enhancement_previewed: previewed
    };
    cachedDecision = { key: cacheKey, value: { ...value } };
    return value;
  }

  async function previewMainPrompt() {
    const button = $('previewEnhance');
    button.disabled = true;
    try {
      const raw = $('prompt').value.trim();
      if (!raw) throw Error('Enter a prompt first.');
      const cfg = enhancementConfig({ enabled: true, show: true });
      message('Generating enhanced prompt preview…');
      const enhanced = await enhanceOnly(raw, cfg.mode, false);
      const choice = await choosePrompt(raw, enhanced, 'generation');
      if (choice === null) {
        message('Prompt preview closed.');
        return;
      }
      const value = {
        original_prompt: raw,
        enhanced_prompt: enhanced,
        used_prompt: choice === 'original' ? raw : enhanced,
        prompt_enhancement_enabled: true,
        prompt_enhancement_mode: cfg.mode,
        prompt_enhancement_previewed: true
      };
      cachedDecision = { key: [raw, cfg.mode, 'show', 'create'].join('\u0000'), value };
      message(choice === 'original' ? 'Original prompt selected for the next generation.' : 'Enhanced prompt selected for the next generation.', 'good');
    } catch (error) {
      message(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  // Replace the main submit path so the phone UI can preview an AI rewrite and
  // archive both the original and the prompt that MiniMax actually receives.
  submitGeneration = async function () {
    if (submitting) return;
    submitting = true;
    $('generate').disabled = true;
    let localId = null;
    try {
      const config = connection();
      const url = $('imageUrl').value.trim();
      const file = url ? null : ($('image').files[0] || savedImageBlob);
      const originalPrompt = $('prompt').value.trim();
      const settings = snapshotSettings();
      const presetName = currentPresetName();
      if (!originalPrompt) throw Error('Enter a prompt.');
      if (!file && !url) throw Error('Choose an image or enter an HTTPS image link.');
      if (file && file.size > 6000000) throw Error('Image exceeds 6 MB.');
      if (url && new URL(url).protocol !== 'https:') throw Error('Image link must use HTTPS.');
      if (!Number.isFinite(settings.duration) || settings.duration < 0 || settings.duration > 60) throw Error('Video length must be 0–60 seconds.');

      const prepared = await preparePrompt(originalPrompt, false);
      if (!prepared) {
        message('Generation cancelled.');
        return;
      }
      safeSet(STORAGE.prompt, originalPrompt);
      safeSet(STORAGE.duration, String(settings.duration));
      if (url) safeSet(STORAGE.imageUrl, url);

      localId = uid();
      jobs.push({
        localId,
        jobId: null,
        endpoint: config.endpoint,
        status: 'submitting',
        stage: 'Submitting',
        progress: 0,
        detail: prepared.prompt_enhancement_enabled ? 'AI prompt ready' : '',
        prompt: prepared.used_prompt,
        originalPrompt: prepared.original_prompt,
        presetName,
        settings,
        createdAt: Date.now()
      });
      persistJobs();
      openDrawer('queue');
      const image = file ? await fileData(file) : url;
      const queued = await request(config, '/run', {
        input: {
          image,
          prompt: prepared.used_prompt,
          length_seconds: settings.duration,
          preset_name: presetName,
          ...runtimeInput(),
          ...prepared
        },
        policy: { executionTimeout: 7200000, ttl: 86400000 }
      });
      if (typeof queued.id !== 'string' || !queued.id) throw Error('RunPod did not return a job ID.');
      if (!findJob(localId)) {
        try { await request(config, '/cancel/' + encodeURIComponent(queued.id), {}); } catch {}
        return;
      }
      updateJob(localId, { jobId: queued.id, status: 'queued', stage: 'Queued', detail: 'Waiting for RunPod', progress: 0 });
      message('Added job to the queue. You can submit another one now.', 'good');
      void pollJob(localId);
    } catch (error) {
      message(error.message, 'error');
      if (localId && findJob(localId)) updateJob(localId, { status: 'failed', stage: 'Submit failed', detail: error.message, progress: 100 });
    } finally {
      submitting = false;
      $('generate').disabled = false;
    }
  };

  function improveDownloadLink(anchor, video) {
    if (!anchor || !video?.url) return;
    anchor.href = video.download_url || video.url;
    anchor.textContent = 'Download MP4';
    anchor.setAttribute('download', video.filename || 'redgraft-video.mp4');
    anchor.target = '_self';
    anchor.rel = 'noopener';
  }

  const baseShowRecentResult = showRecentResult;
  showRecentResult = function (output, job) {
    baseShowRecentResult(output, job);
    const cards = [...document.querySelectorAll('#results .result-card')];
    const videos = Array.isArray(output?.videos) ? output.videos : [];
    cards.forEach((card, index) => improveDownloadLink(card.querySelector('a.download'), videos[index]));
  };

  const baseRenderLibrary = renderLibrary;
  renderLibrary = function () {
    baseRenderLibrary();
    const cards = [...document.querySelectorAll('#videoLibrary .video-card')];
    cards.forEach((card, index) => {
      const video = libraryItems[index]?.videos?.[0];
      improveDownloadLink(card.querySelector('a.download'), video);
      if (card.querySelector('.mobile-download-hint') || !video?.download_url) return;
      const hint = document.createElement('div');
      hint.className = 'tiny mobile-download-hint';
      hint.textContent = 'Mobile: tap Download MP4 for a file-download response instead of opening the player.';
      card.append(hint);
    });
  };

  const baseOpenDetails = openDetails;
  openDetails = function (render) {
    baseOpenDetails(render);
    const original = render.original_prompt || render.prompt || '';
    const enhanced = render.enhanced_prompt || '';
    const used = render.used_prompt || render.prompt || '';
    $('detailPrompt').textContent = used;
    $('detailOriginalPrompt').textContent = original;
    $('detailEnhancedPrompt').textContent = enhanced || 'Not used';
    $('detailOriginalWrap').hidden = !original;
    $('detailEnhancedWrap').hidden = !enhanced;
    const meta = $('detailMeta');
    const add = (label, value) => {
      if (value === undefined || value === null || value === '') return;
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = String(value);
      meta.append(dt, dd);
    };
    add('Enhancement', render.prompt_enhancement?.enabled ? (render.prompt_enhancement.mode || 'on') : 'off');
    if (render.extension?.from_render_id) add('Extended from', render.extension.from_render_id);
    if (render.extension) add('Merged video', render.extension.merged ? 'yes' : 'no');
  };

  $('autoEnhance').checked = (safeGet(ENHANCE_STORAGE.enabled) || '1') === '1';
  $('enhanceMode').value = safeGet(ENHANCE_STORAGE.mode) || 'detailed';
  $('showEnhancedPrompt').checked = (safeGet(ENHANCE_STORAGE.show) || '1') === '1';
  $('autoEnhance').addEventListener('change', () => { cachedDecision = null; persistEnhancementConfig(); });
  $('enhanceMode').addEventListener('change', () => { cachedDecision = null; persistEnhancementConfig(); });
  $('showEnhancedPrompt').addEventListener('change', () => { cachedDecision = null; persistEnhancementConfig(); });
  $('prompt').addEventListener('input', () => { cachedDecision = null; });
  $('previewEnhance').addEventListener('click', () => void previewMainPrompt());
  $('useEnhancedPrompt').addEventListener('click', () => closePromptChoice('enhanced'));
  $('useOriginalPrompt').addEventListener('click', () => closePromptChoice('original'));
  $('cancelPromptChoice').addEventListener('click', () => closePromptChoice(null));
  $('promptPreviewModal').addEventListener('click', event => { if (event.target === $('promptPreviewModal')) closePromptChoice(null); });

  window.RedgraftPromptEnhancer = {
    prepare: preparePrompt,
    enhanceOnly,
    config: enhancementConfig,
    persisted: () => ({
      enabled: $('autoEnhance').checked,
      mode: $('enhanceMode').value,
      show: $('showEnhancedPrompt').checked
    })
  };

  // app.js rendered the cached library before this file loaded.
  renderLibrary();
})();
