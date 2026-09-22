'use strict';
// The production endpoint may still be running the previous strict worker while
// the new archive-capable image is being rolled out. Strip the UI-only preset
// label from generation requests so queueing keeps working on both versions.
const redgraftBaseRequest=request;
request=(config,path,body)=>{
  let next=body;
  if(path==='/run'&&body?.input&&body.input.action===undefined&&Object.prototype.hasOwnProperty.call(body.input,'preset_name')){
    next={...body,input:{...body.input}};
    delete next.input.preset_name;
  }
  return redgraftBaseRequest(config,path,next);
};

const openQueueButton=document.getElementById('openQueue');
if(openQueueButton)openQueueButton.addEventListener('click',()=>document.getElementById('menuBtn')?.click());
const videosTab=document.getElementById('tabVideos');
if(videosTab)videosTab.addEventListener('click',()=>{if(typeof refreshLibrary==='function')void refreshLibrary(true)});
window.addEventListener('online',()=>{if(typeof jobs!=='undefined'&&typeof jobActive==='function'&&typeof pollJob==='function'){for(const job of jobs.filter(jobActive))if(job.jobId)void pollJob(job.localId)}});
