import React from 'react';
import { Button, TextField, CircularProgress } from '@material-ui/core'
import './index.css';

class Page extends React.Component{
    constructor(props){
        super(props);
        this.updateState = this.updateState.bind(this);
        this.clearState = this.clearState.bind(this);
        this.state = {text: '待转换文本内容', file_path: ''};
        
    }
    renderMainTitle() {
        return (
        <div className="maintitel-div">
            <p><font size="8">Text to Speech System</font></p>
            <p>based on React & Tensorflow.js by Purkialo</p>
        </div>
        );
    }

    updateState(){
        this.setState({text: document.getElementById("input_text").value, file_path: ""});
        if (document.getElementById("input_text").value != ""){
            var formData = new FormData();
            formData.append('tts_text', document.getElementById("input_text").value);
            fetch("/tts_text", {method : 'POST', body : formData})
                .then(result => result.json())
                .then(data => {
                    this.setState({file_path: data.file_path});
                });
        }
    }

    clearState(){
        this.setState({text: '待转换文本内容'});
        document.getElementById("input_text").value = '';
    }

    renderProgressAudio(){
        if(this.state.text != '待转换文本内容' && this.state.text != '' && this.state.file_path === ''){
            return (<CircularProgress />);
        }
        if(this.state.file_path != ''){
            return (
                <div>
                <audio src={this.state.file_path} controls="controls">
                    Your browser does not support the audio element.
                </audio>
                </div>
            )
        }
    }
    
    render(){
		return(
            <div>
                <div className="titel-div">{this.renderMainTitle()}</div>
                <div className="textarea-div">
                    <TextField 
                        id="input_text" placeholder="待转换文本内容" 
                        multiline="true" variant="outlined" rows="5" fullWidth='true'
                        error={this.state.text === ""} 
                        helperText={this.state.text === ""?"待转换文本不能为空":" "}>
                    </TextField>
                    <table className="table-button">
                        <td>
                            <Button className="buttons" variant="contained" color="primary" component="submit" disableElevation onClick={this.updateState}>
                                开始转换
                            </Button>
                        </td>
                        <td>
                            <Button className="buttons" variant="contained" color="default" component="button" disableElevation onClick={this.clearState}>
                                清空
                            </Button>
                        </td>
                    </table>
                </div>
                <div className="progress-audio">
                    {this.renderProgressAudio()}
                </div>
            </div>
            
		)
	}
}
export default Page;