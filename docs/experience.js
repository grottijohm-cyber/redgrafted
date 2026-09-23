'use strict';
(() => {
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
      hint.textContent = 'On mobile, Download MP4 saves the file directly instead of opening the player.';
      card.append(hint);
    });
  };

  const baseOpenDetails = openDetails;
  openDetails = function (render) {
    baseOpenDetails(render);
    if (render.extension?.from_render_id) {
      const meta = $('detailMeta');
      const add = (label, value) => {
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        dt.textContent = label;
        dd.textContent = String(value);
        meta.append(dt, dd);
      };
      add('Extended from', render.extension.from_render_id);
      add('Merged video', render.extension.merged ? 'yes' : 'no');
    }
  };

  renderLibrary();
})();
