import {z} from 'zod';
import {AppError,imageSchema,profileOf,scoreSchema,readBody,errorResponse} from '@/lib/health';

const modalitySchema=z.enum(['chest_xray','brain_mri']);
const findingSchema=z.object({label:z.string(),score:z.number(),confidence:z.string(),note:z.string()});
const xrayResponseSchema=z.object({
 image_type:z.literal('chest_xray'),image_quality:z.string(),status:z.string(),model:z.string(),device:z.enum(['cpu','cuda']),
 model_scores:z.record(z.number()),highest_scoring_labels:z.array(findingSchema),integrated_summary:z.string(),limitations:z.array(z.string()),
 clinical_review_required:z.boolean(),disclaimer:z.string(),
});
const mriResponseSchema=z.object({
 request_id:z.string().optional(),model:z.string(),device:z.enum(['cpu','cuda']),input_type:z.literal('brain_mri_2d_image'),
 predicted_label:z.enum(['glioma','meningioma','no_tumor','pituitary']),class_scores:z.record(z.number()),
 ranked_labels:z.array(z.object({label:z.string(),score:z.number()})),integrated_summary:z.string(),
 clinical_review_required:z.boolean(),disclaimer:z.string(),
});

function riskLevel(score:number){return score>=67?'high':score>=34?'moderate':'low'}

export async function POST(req:Request){
 let modality:'chest_xray'|'brain_mri'='chest_xray';
 try{
  const body=await readBody(req);
  modality=modalitySchema.parse(body.modality??'chest_xray');
  const profile=profileOf(body.profile),scores=scoreSchema.parse(body.assessment?.scores),image=imageSchema.parse(body.image);
  const prefix=`data:${image.mimeType};base64,`;
  if(!image.data.startsWith(prefix))throw new AppError('Invalid medical image encoding.','INVALID_IMAGE',400,false);
  const healthReport={source:'vitalis',overall_health_score:body.assessment?.overall,summary:typeof body.summary==='string'?body.summary.slice(0,1200):null,risk_signals:Object.entries(scores).map(([name,score])=>({model_name:name,label:name.replaceAll('_',' '),risk_level:riskLevel(score),score:score/100})),profile_context:{age:profile.age,gender:profile.gender,bmi:profile.bmi}};

  if(modality==='brain_mri'){
   const endpoint=process.env.MRI_API_URL||'http://127.0.0.1:8003/mri';
   let response:Response;
   try{response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.MODEL_API_KEY?{Authorization:`Bearer ${process.env.MODEL_API_KEY}`}:{})},body:JSON.stringify({mime_type:image.mimeType,image_base64:image.data.slice(prefix.length),health_report:healthReport}),signal:AbortSignal.timeout(60000)})}catch(error){if(error instanceof DOMException&&error.name==='TimeoutError')throw new AppError('The brain MRI model timed out. Retry shortly.','MRI_TIMEOUT',504,true);throw new AppError('Could not reach the brain MRI service. Verify MRI_API_URL and try again.','MRI_SERVICE_UNAVAILABLE',503,true,error instanceof Error?error.message:undefined)}
   if(!response.ok){let detail='';try{detail=(await response.json() as {detail?:string}).detail||''}catch{}throw new AppError(detail||'The brain MRI model could not analyze this image.','MRI_ANALYSIS_FAILED',response.status>=500?503:response.status,response.status>=500,`MRI service HTTP ${response.status}`)}
   const output=mriResponseSchema.parse(await response.json());
   return Response.json({modality,summary:output.integrated_summary,observations:output.ranked_labels.map(item=>`${item.label.replaceAll('_',' ')}: ${(item.score*100).toFixed(1)}% model score`),limitations:'This classifier accepts exported 2D brain MRI images only. It does not inspect a full MRI series, confirm that the image is an MRI, or replace review of the original study.',model:output.model,device:output.device,modelScores:output.class_scores,topPrediction:output.predicted_label,clinicalReviewRequired:output.clinical_review_required,disclaimer:output.disclaimer});
  }

  const modelBase=(process.env.MODEL_API_URL||'http://127.0.0.1:8000/predict').replace(/\/predict\/?$/,'');
  const endpoint=process.env.XRAY_API_URL||`${modelBase}/xray`;
  let response:Response;
  try{response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.MODEL_API_KEY?{Authorization:`Bearer ${process.env.MODEL_API_KEY}`}:{})},body:JSON.stringify({mime_type:image.mimeType,image_base64:image.data.slice(prefix.length),health_report:healthReport}),signal:AbortSignal.timeout(60000)})}catch(error){if(error instanceof DOMException&&error.name==='TimeoutError')throw new AppError('The X-ray model timed out. Its first run may still be downloading the model checkpoint; retry shortly.','XRAY_TIMEOUT',504,true);throw new AppError('Could not reach the X-ray model service. Start the Python model server and try again.','XRAY_SERVICE_UNAVAILABLE',503,true,error instanceof Error?error.message:undefined)}
  if(!response.ok){let detail='';try{detail=(await response.json() as {detail?:string}).detail||''}catch{}throw new AppError(detail||'The X-ray model could not analyze this image.','XRAY_ANALYSIS_FAILED',response.status>=500?503:response.status,response.status>=500,`Model service HTTP ${response.status}`)}
  const output=xrayResponseSchema.parse(await response.json());
  return Response.json({modality,summary:output.integrated_summary,observations:output.highest_scoring_labels.map(item=>`${item.label}: model score ${item.score.toFixed(3)} (${item.note})`),limitations:output.limitations.join(' '),model:output.model,device:output.device,modelScores:output.model_scores,topFindings:output.highest_scoring_labels,clinicalReviewRequired:output.clinical_review_required,disclaimer:output.disclaimer});
 }catch(error){return errorResponse(error,modality==='brain_mri'?'brain MRI analysis':'chest X-ray analysis')}
}
