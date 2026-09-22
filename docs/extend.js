'use strict';
(() => {
  const RUNTIME_KEYS = [
    'turbo_strength', 'm3_strength', 'mystic_strength', 'hmnsfw_strength',
    'vagassist_strength', 'hmpussy_strength', 'cumshot_strength', 'steps',
    'enable_audio', 'enable_gimm'
  ];

  const baseRenderLibrary = renderLibrary;

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

  async function extendRender(render, button) {
    if (!render?.render_id) {
      message('This render does not have a permanent render ID.', 'error');
      return;
    }

    const promptValue = window.prompt(
      'Prompt for the continuation segment:',
      String(render.prompt || '')
    );
    if (promptValue === null) return;
    const continuationPrompt = promptValue.trim();
    if (!continuationPrompt) {
      message('Enter a prompt for the continuation.', 'error');
      return;
    }

    const secondsValue = window.prompt(
      'How many seconds should be added? (1–60)',
      String(extensionDuration(render))
    );
    if (secondsValue === null) return;
    const seconds = Number(secondsValue);
    if (!Number.isFinite(seconds) || seconds < 1 || seconds > 60) {
      message('Extension length must be between 1 and 60 seconds.', 'error');
      return;
    }

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
        detail: 'Using the previous video’s final frame',
        prompt: continuationPrompt,
        presetName: (render.preset_name || 'Custom') + ' · Extend',
        settings: { duration: seconds, ...runtime },
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
          prompt: continuationPrompt,
          length_seconds: seconds,
          ...runtime
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
        detail: 'Final frame will seed the continuation',
        progress: 0
      });
      message('Video extension added to the queue.', 'good');
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
      button.title = 'Continue this video from its final frame';
      button.addEventListener('click', () => void extendRender(render, button));
      actions.prepend(button);
    });
  }

  renderLibrary = function () {
    baseRenderLibrary();
    addExtendButtons();
  };

  // Handles the case where app.js finished its initial render before this file loaded.
  addExtendButtons();
})();
