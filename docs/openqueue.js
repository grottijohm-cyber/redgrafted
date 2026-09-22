'use strict';
const openQueueButton=document.getElementById('openQueue');
if(openQueueButton)openQueueButton.addEventListener('click',()=>document.getElementById('menuBtn')?.click());
const videosTab=document.getElementById('tabVideos');
if(videosTab)videosTab.addEventListener('click',()=>{if(typeof refreshLibrary==='function')void refreshLibrary(true)});
window.addEventListener('online',()=>{if(typeof jobs!=='undefined'&&typeof jobActive==='function'&&typeof pollJob==='function'){for(const job of jobs.filter(jobActive))if(job.jobId)void pollJob(job.localId)}});
