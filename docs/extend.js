'use strict';
(() => {
  const RUNTIME_KEYS = [
    'turbo_strength', 'm3_strength', 'mystic_strength', 'hmnsfw_strength',
    'vagassist_strength', 'hmpussy_strength', 'cumshot_strength',
    'realism_strength', 'deepthroat_strength', 'civ3210503_strength', 'civ3320641_strength',
    'pussy4nus_strength', 'fingering_strength', 'moawxx_strength', 'naughtytimes_strength',
    'steps', 'enable_audio', 'enable_gimm', 'enable_ai_upscale'
  ];
  const baseRenderLibrary = renderLibrary;
  let extendResolver = null;

  function extensionDuration(render) {
    const value = Number(render?.request_settings?.length_seconds);
    return Number.isFinite(value) && value >= 1 && value <= 60 ? Math.round(value) : 8;
  }

  function inheritedRuntime(render) {
    const settings = render?.request_settings && typeof render.request_settings === 'object'
      ? render.request_settings
      : {};
    const out = {};
    for (const key of RUNTIME_KEYS) {
      if (Object.prototype.hasOwnProperty.call(settings, key)) out[key] = settings[key];
    }
    return out;
  }

  function closeExtendDialog(value) {
    $('extendModal').hidden = true;
    const resolve = extendResolver;
    extendResolver = null;
    if (resolve) resolve(value);
  }

  function askExtension(render) {
    $('extendPrompt').value = String(render.used_prompt || render.prompt || 'Continue the motion naturally and seamlessly.');
    $('extendDuration').value = String(extensionDuration(render));
    $('extendModal').hidden = false;
    return new Promise(resolve => { extendResolver = resolve; });
  }

  function submitExtendDialog() {
    const prompt = $('extendPrompt').value.trim();
    const seconds = Number($('extendDuration').value);
    if (!prompt) {
      message('Enter a continuation prompt.', 'error');
      return;
    }
    if (!Number.isFinite(seconds) || seconds < 1 || seconds > 60) {
      message('Extension length must be between 1 and 60 seconds.', 'error');
      return;
    }
    closeExtendDialog({ prompt, seconds });
  }

  async function extendRender(render, button) {
    if (!render?.render_id) {
      message('This render does not have a permanent render ID.', 'error');
      return;
    }
    const options = await askExtension(render);
    if (!options) return;

    const prepared = {
      original_prompt: options.prompt,
      enhanced_prompt: null,
      used_prompt: options.prompt,
      prompt_enhancement_enabled: false,
      prompt_enhancement_mode: null,
      prompt_enhancement_previewed: false
    };

    let localId = null;
    try {
      const config = connection();
      const runtime = inheritedRuntime(render);
      localId = uid();
      const job = {
        localId,
        jobId: null,
        endpoint: config.endpoint,
        status: 'submitting',
        stage: 'Preparing extension',
        progress: 0,
        detail: 'Final frame → continuation → append to original',
        prompt: prepared.used_prompt,
        originalPrompt: prepared.original_prompt,
        presetName: (render.preset_name || 'Custom') + ' · Extend',
        settings: { duration: options.seconds, ...runtime },
        sourceRenderId: render.render_id,
        createdAt: Date.now()
      };
      jobs.push(job);
      persistJobs();
      openDrawer('queue');
      if (button) button.disabled = true;

      const queued = await request(config, '/run', {
        input: {
          action: 'extend',
          render_id: render.render_id,
          prompt: prepared.used_prompt,
          length_seconds: options.seconds,
          ...runtime,
          ...prepared
        },
        policy: { executionTimeout: 7200000, ttl: 86400000 }
      });
      if (typeof queued.id !== 'string' || !queued.id) {
        throw Error('RunPod did not return an extension job ID.');
      }
      if (!findJob(localId)) {
        try { await request(config, '/cancel/' + encodeURIComponent(queued.id), {}); } catch {}
        return;
      }
      updateJob(localId, {
        jobId: queued.id,
        status: 'queued',
        stage: 'Queued extension',
        detail: 'Will append the continuation into one MP4',
        progress: 0
      });
      message('Video extension added to the queue. The finished result will include the original + continuation.', 'good');
      void pollJob(localId);
    } catch (error) {
      message(error.message, 'error');
      if (localId && findJob(localId)) {
        updateJob(localId, {
          status: 'failed',
          stage: 'Extension failed',
          detail: error.message,
          progress: 100
        });
      }
    } finally {
      if (button) button.disabled = false;
    }
  }

  function addExtendButtons() {
    const cards = [...document.querySelectorAll('#videoLibrary .video-card')];
    cards.forEach((card, index) => {
      if (card.querySelector('[data-extend-render]')) return;
      const render = libraryItems[index];
      if (!render?.render_id || !render?.videos?.[0]) return;
      const actions = card.querySelector('.video-actions');
      if (!actions) return;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'secondary';
      button.textContent = 'Extend';
      button.dataset.extendRender = render.render_id;
      button.title = 'Continue from the final frame and append it to this video';
      button.addEventListener('click', () => void extendRender(render, button));
      actions.prepend(button);
    });
  }

  $('queueExtend').addEventListener('click', submitExtendDialog);
  $('cancelExtend').addEventListener('click', () => closeExtendDialog(null));
  $('closeExtend').addEventListener('click', () => closeExtendDialog(null));
  $('extendModal').addEventListener('click', event => { if (event.target === $('extendModal')) closeExtendDialog(null); });

  renderLibrary = function () {
    baseRenderLibrary();
    addExtendButtons();
  };
  addExtendButtons();
})();
