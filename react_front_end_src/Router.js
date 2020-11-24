import React from 'react';
import {BrowserRouter, Route, Redirect} from 'react-router-dom'
import Page from './Page';
//<Route path="/tts_trans" component={Page} />

function Router(){
    return (
        <BrowserRouter>
            <Route path="/tts_trans" render={
                () => (
                    <Page />
                ) 
            } />
            <Route path="/" render={
                () => (
                    <Redirect to="/tts_trans" />
                )
            } />
        </BrowserRouter>
    );
}

export default Router;