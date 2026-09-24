'use strict';
// The production endpoint may still be running the previous strict worker while
// the new archive-capable image is being rolled out. Strip the UI-only preset
// label from generation requests so queueing keeps working on both versions.
const redgraftBaseRequest=request;
request=(config,path,body)=>{
  let next=body;
  if(path==='/run'&&body?.input&&body.input.action===undefined){
    next={...body,input:{...body.input}};
    // UI-only metadata is handled by the archive wrapper, not the base generator.
    delete next.input.preset_name;
    // These controls were added after the first prompt-enhancer worker. Keeping
    // their default values out of the request lets the updated phone UI keep
    // generating against that worker while a new image is being rolled out.
    for(const key of [
      'realism_strength','deepthroat_strength','civ3210503_strength','civ3320641_strength',
      'pussy4nus_strength','fingering_strength','moawxx_strength','naughtytimes_strength'
    ]){
      if(Number(next.input[key]||0)===0) delete next.input[key];
    }
    if(next.input.enable_ai_upscale!==true) delete next.input.enable_ai_upscale;
    // Preserve ordinary I2V compatibility while the Ref2VA worker rolls out.
    if(next.input.generation_mode==='i2v') delete next.input.generation_mode;
    if(next.input.quality_mode==='fast') delete next.input.quality_mode;
    if(Array.isArray(next.input.reference_images)&&next.input.reference_images.length===0) delete next.input.reference_images;
    if(next.input.quality_mode==='fast') delete next.input.quality_mode;
    // I2V is the worker default. Omit the explicit mode so the refreshed UI can
    // still submit ordinary generations to the immediately previous worker.
    if(next.input.generation_mode==='i2v') delete next.input.generation_mode;
  }
  return redgraftBaseRequest(config,path,next);
};

const openQueueButton=document.getElementById('openQueue');
if(openQueueButton)openQueueButton.addEventListener('click',()=>document.getElementById('menuBtn')?.click());
const videosTab=document.getElementById('tabVideos');
if(videosTab)videosTab.addEventListener('click',()=>{if(typeof refreshLibrary==='function')void refreshLibrary(true)});
window.addEventListener('online',()=>{if(typeof jobs!=='undefined'&&typeof jobActive==='function'&&typeof pollJob==='function'){for(const job of jobs.filter(jobActive))if(job.jobId)void pollJob(job.localId)}});
